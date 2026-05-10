{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (nixpkgs) lib;
      in
      {
        devShells.default = with pkgs; mkShell {
          packages =
          [
            bashInteractive
            uv
            nodejs
            python312Packages.pycairo
            stdenv.cc.cc.lib 
          ];

          # env = lib.optionalAttrs pkgs.stdenv.isLinux {
          #     # Python libraries often load native shared objects using dlopen(3).                                          
          #   # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
          #   # 
          #     LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          #   };

          shellHook = ''
              export LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"                                               
            '';
                                       
         };
      }
    );
}
