{
  description = "krr with nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
          };
        in
        {
          devShell = pkgs.mkShell
            {
              buildInputs = with pkgs; [
                stdenv.cc.cc.lib
              ];
              shellHook = ''
                LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/
              '';
            };
        });
}
