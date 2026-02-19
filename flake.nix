{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rust = pkgs.rust-bin.selectLatestNightlyWith (
          toolchain:
          toolchain.default.override {
            extensions = [
              "rust-src"
              "llvm-tools-preview"
            ];
          }
        );
      in
      {
        devShell = pkgs.mkShell {
          buildInputs =
            lib.singleton rust
            ++ (with pkgs; [
              cargo-nextest
              cargo-llvm-cov
              cargo-criterion
              cargo-pgo
              llvmPackages.bolt
              gnuplot
              taplo
            ]);
        };
      }
    );
}
