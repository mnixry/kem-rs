{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    { flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      imports = [ inputs.treefmt-nix.flakeModule ];
      perSystem =
        { lib, system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ (import inputs.rust-overlay) ];
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
          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (_: rust);
        in
        {
          _module.args = { inherit inputs pkgs rust; };
          treefmt =
            { ... }:
            {
              projectRootFile = ".git/config";
              programs.nixfmt.enable = true;
              programs.taplo.enable = true;
              programs.rustfmt = {
                enable = true;
                package = rust;
              };
              programs.yamlfmt.enable = true;
            };
          devShells.default = pkgs.mkShell {
            buildInputs =
              lib.singleton rust
              ++ (with pkgs; [
                cargo-edit
                cargo-nextest
                cargo-llvm-cov
                cargo-criterion
                cargo-pgo
                llvmPackages.bolt
                gnuplot
                taplo
              ]);
          };
          checks =
            let
              commonArgs = (craneLib.crateNameFromCargoToml { cargoToml = ./crates/lib/Cargo.toml; }) // rec {
                src = ./.;
                strictDeps = true;
                cargoVendorDir = craneLib.vendorMultipleCargoDeps {
                  inherit (craneLib.findCargoFiles src) cargoConfigs;
                  cargoLockList = [
                    ./Cargo.lock
                    "${rust}/lib/rustlib/src/rust/library/Cargo.lock"
                  ];
                };
              };
              cargoArtifacts = craneLib.buildDepsOnly commonArgs;
            in
            {
              clippy = craneLib.cargoClippy (
                commonArgs
                // {
                  inherit cargoArtifacts;
                  cargoClippyExtraArgs = "--all-targets --all-features";
                }
              );
              test = craneLib.cargoNextest (
                commonArgs
                // {
                  inherit cargoArtifacts;
                }
              );
            };
        };
    };
}
