#!/usr/bin/env python3
"""ACE-Step Multi-GPU Debug Notebook.

Runs on Kaggle with T4x2 to diagnose multi-GPU model distribution.
Workflow:
  1. Clone accel branch from voland62/ACE-Step-1.5
  2. Install deps via uv
  3. Symlink checkpoints from Kaggle dataset
  4. Start acestep-api as background process, capture logs
  5. Wait for server readiness
  6. Run integration tests (release_task + query_result)
  7. Capture nvidia-smi diagnostics
  8. Save all logs to /kaggle/working/ for download
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

# ===========================================================================
# Configuration
# ===========================================================================

REPO_URL = "https://github.com/voland62/ACE-Step-1.5.git"
BRANCH = "accel"
WORK_DIR = "/kaggle/working/ACE-Step-1.5"
CHECKPOINT_DATASET = "/kaggle/input/datasets/andrewvoron/ace-chekpoints"
CHECKPOINT_DIR = f"{WORK_DIR}/checkpoints"
API_HOST = "127.0.0.1"
API_PORT = 8001
API_BASE = f"http://{API_HOST}:{API_PORT}"
LOG_DIR = "/kaggle/working/logs"

# Test configs: turbo (small, fast) and xs-base (to test inter-model placement)
TEST_CONFIGS = [
    {
        "name": "turbo",
        "config_path": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-0.6B",
    },
]


def log(msg):
    """Print with timestamp for log readability."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, **kwargs):
    """Run a shell command, printing it first."""
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, **kwargs)


# ===========================================================================
# Step 1: Clone repository
# ===========================================================================

log("=" * 70)
log("STEP 1: Clone repository")
log("=" * 70)

os.makedirs(LOG_DIR, exist_ok=True)

if os.path.exists(WORK_DIR):
    log(f"Removing existing {WORK_DIR}")
    shutil.rmtree(WORK_DIR)

run(f"git clone --branch {BRANCH} --depth 1 {REPO_URL} {WORK_DIR}")
os.chdir(WORK_DIR)
run("git log --oneline -1")

# ===========================================================================
# Step 2: Install dependencies
# ===========================================================================

log("=" * 70)
log("STEP 2: Install dependencies")
log("=" * 70)

run('curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh')
run("uv sync")

# ===========================================================================
# Step 3: Setup checkpoints (hybrid symlink structure)
# ===========================================================================

log("=" * 70)
log("STEP 3: Setup checkpoints")
log("=" * 70)

ACE_INNER_DIR = os.path.join(CHECKPOINT_DATASET, "Ace-Step1.5")

if os.path.exists(CHECKPOINT_DIR):
    shutil.rmtree(CHECKPOINT_DIR)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def smart_link_or_copy(source_path, target_path):
    """Symlink heavy files (safetensors, bin), copy configs."""
    if os.path.isdir(source_path):
        os.makedirs(target_path, exist_ok=True)
        for item in os.listdir(source_path):
            smart_link_or_copy(
                os.path.join(source_path, item),
                os.path.join(target_path, item),
            )
    else:
        if source_path.endswith((".safetensors", ".bin", ".pt", ".pth")):
            if not os.path.exists(target_path):
                os.symlink(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


# Process root of dataset
for item in glob.glob(f"{CHECKPOINT_DATASET}/*"):
    name = os.path.basename(item)
    if name == "Ace-Step1.5":
        continue
    smart_link_or_copy(item, os.path.join(CHECKPOINT_DIR, name))

# Process inner Ace-Step1.5 directory
if os.path.exists(ACE_INNER_DIR):
    for item in glob.glob(f"{ACE_INNER_DIR}/*"):
        name = os.path.basename(item)
        smart_link_or_copy(item, os.path.join(CHECKPOINT_DIR, name))

log("Checkpoint structure:")
run(f"ls -la {CHECKPOINT_DIR}/")
run(f"find {CHECKPOINT_DIR}/ -maxdepth 2 -type d")

# ===========================================================================
# Step 4: GPU diagnostics (before server start)
# ===========================================================================

log("=" * 70)
log("STEP 4: GPU diagnostics")
log("=" * 70)

run("nvidia-smi")
run("nvidia-smi -L")

# Python-level GPU detection
gpu_diag_script = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / (1024**3)
    print(f"  GPU {i}: {props.name}, {mem_gb:.1f} GB")
"""
run(f'uv run python3 -c "{gpu_diag_script}"')

# ===========================================================================
# Step 5: Write .env and start API server
# ===========================================================================

log("=" * 70)
log("STEP 5: Start API server")
log("=" * 70)

env_text = f"""
ACESTEP_CHECKPOINTS_DIR={CHECKPOINT_DIR}
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_VAE_CHECKPOINT=scragvae
ACESTEP_INIT_LLM=true
ACESTEP_NO_INIT=false
ACESTEP_BATCH_SIZE=1
ACESTEP_LM_BACKEND=pt
ACESTEP_DTYPE=float16
MPLBACKEND=agg
HF_HOME=/kaggle/working/hf_cache
TRITON_CACHE_DIR=/kaggle/working/triton_cache
PYTORCH_ALLOC_CONF=expandable_segments:True
"""

with open(f"{WORK_DIR}/.env", "w") as f:
    f.write(env_text.strip() + "\n")
log(".env written")

# Start server as background process, capture all output
api_log_path = f"{LOG_DIR}/api_server.log"
log(f"Starting API server (logs → {api_log_path})")

api_log_file = open(api_log_path, "w")
api_process = subprocess.Popen(
    ["uv", "run", "acestep-api"],
    cwd=WORK_DIR,
    stdout=api_log_file,
    stderr=subprocess.STDOUT,
    env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
)
log(f"API server started (PID: {api_process.pid})")

# ===========================================================================
# Step 6: Wait for server readiness
# ===========================================================================

log("=" * 70)
log("STEP 6: Wait for server readiness")
log("=" * 70)

import urllib.request
import urllib.error

MAX_WAIT = 300  # 5 minutes
POLL_INTERVAL = 5
start_time = time.time()
server_ready = False

while time.time() - start_time < MAX_WAIT:
    # Check if process crashed
    if api_process.poll() is not None:
        log(f"API server process exited with code {api_process.returncode}")
        break

    try:
        req = urllib.request.Request(f"{API_BASE}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
            log(f"Health check response: {json.dumps(body, indent=2)}")
            server_ready = True
            break
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        elapsed = int(time.time() - start_time)
        log(f"Waiting for server... ({elapsed}s)")
        time.sleep(POLL_INTERVAL)

if not server_ready:
    log("SERVER FAILED TO START!")
    # Dump what we have in the log
    api_log_file.flush()
    with open(api_log_path) as f:
        log("=== API Server Log (last 200 lines) ===")
        lines = f.readlines()
        for line in lines[-200:]:
            print(line, end="", flush=True)

# Capture nvidia-smi after model loading
nvidia_post_load = f"{LOG_DIR}/nvidia_smi_post_load.txt"
run(f"nvidia-smi > {nvidia_post_load} 2>&1")
with open(nvidia_post_load) as f:
    log("=== nvidia-smi after model load ===")
    print(f.read(), flush=True)

# ===========================================================================
# Step 7: Integration tests
# ===========================================================================

log("=" * 70)
log("STEP 7: Integration tests")
log("=" * 70)

test_results = {}

if server_ready:
    # Test 1: Simple music generation via release_task
    log("--- Test 1: release_task (text2music, turbo) ---")

    release_task_payload = {
        "prompt": "A short electronic beat for testing",
        "lyrics": "",
        "audio_duration": 10,
        "inference_steps": 4,
        "batch_size": 1,
        "task_type": "text2music",
        "thinking": False,
        "use_random_seed": True,
        "audio_format": "wav",
    }

    try:
        req_data = json.dumps(release_task_payload).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE}/release_task",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            release_resp = json.loads(resp.read())
            log(f"release_task response: {json.dumps(release_resp, indent=2)}")
            test_results["release_task"] = release_resp
    except Exception as e:
        log(f"release_task FAILED: {e}")
        test_results["release_task"] = {"error": str(e)}

    # Extract task_id for polling
    task_id = None
    if "data" in test_results.get("release_task", {}):
        data = test_results["release_task"]["data"]
        if isinstance(data, dict):
            task_id = data.get("task_id")
        elif isinstance(data, str):
            task_id = data

    if task_id:
        log(f"Got task_id: {task_id}")

        # Test 2: Poll query_result until completion
        log("--- Test 2: query_result polling ---")
        POLL_MAX = 180  # 3 minutes max for generation
        poll_start = time.time()
        final_result = None

        while time.time() - poll_start < POLL_MAX:
            try:
                query_payload = json.dumps(
                    {"task_id_list": json.dumps([task_id])}
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{API_BASE}/query_result",
                    data=query_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    query_resp = json.loads(resp.read())

                # Check status from response
                data_list = query_resp.get("data", [])
                if data_list and isinstance(data_list, list) and len(data_list) > 0:
                    item = data_list[0]
                    status = item.get("status", -1)
                    log(f"  Poll: status={status}")

                    # status 2 = completed, status 3 = failed
                    if status == 2:
                        log("Generation COMPLETED!")
                        final_result = item
                        break
                    elif status == 3:
                        log(f"Generation FAILED: {item}")
                        final_result = item
                        break
            except Exception as e:
                log(f"  Poll error: {e}")

            time.sleep(5)

        if final_result:
            test_results["query_result_final"] = final_result
            log(f"Final result: {json.dumps(final_result, indent=2)[:2000]}")
        else:
            log("Generation timed out!")
            test_results["query_result_final"] = {"error": "timeout"}

    # Capture nvidia-smi during/after generation
    nvidia_gen = f"{LOG_DIR}/nvidia_smi_after_generation.txt"
    run(f"nvidia-smi > {nvidia_gen} 2>&1")
    with open(nvidia_gen) as f:
        log("=== nvidia-smi after generation ===")
        print(f.read(), flush=True)
else:
    log("SKIPPING integration tests (server not ready)")

# ===========================================================================
# Step 8: Save all diagnostics
# ===========================================================================

log("=" * 70)
log("STEP 8: Save diagnostics")
log("=" * 70)

# Flush and close API log
api_log_file.flush()

# Stop the server gracefully
log("Stopping API server...")
api_process.terminate()
try:
    api_process.wait(timeout=15)
except subprocess.TimeoutExpired:
    api_process.kill()

# Close the log file after server stops
api_log_file.close()

# Save test results
results_path = f"{LOG_DIR}/test_results.json"
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2, default=str)
log(f"Test results saved to {results_path}")

# Copy api log to working directory root for easy access
shutil.copy2(api_log_path, "/kaggle/working/api_server.log")

# Print the full API server log
log("=" * 70)
log("FULL API SERVER LOG")
log("=" * 70)
with open(api_log_path) as f:
    content = f.read()
    print(content, flush=True)
    log(f"API log size: {len(content)} bytes, {content.count(chr(10))} lines")

# Final summary
log("=" * 70)
log("SUMMARY")
log("=" * 70)
log(f"Server started: {'YES' if server_ready else 'NO'}")
log(f"Process exit code: {api_process.returncode}")
log(f"Test results: {list(test_results.keys())}")
for k, v in test_results.items():
    status = "OK" if "error" not in v else f"FAIL: {v.get('error', '')[:100]}"
    log(f"  {k}: {status}")

log("Debug notebook complete.")
