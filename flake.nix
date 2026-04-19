{
  inputs = {
    self.submodules = true;
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
        {
          config,
          lib,
          system,
          ...
        }:
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
          craneCommonArgs =
            (craneLib.crateNameFromCargoToml { cargoToml = ./crates/lib/Cargo.toml; })
            // (
              let
                targetCpu = pkgs.runCommandLocal "rust-current-target-cpu" { buildInputs = [ rust ]; } ''
                  rustc --print target-cpus \
                    | sed -nE 's/.*native.*\(currently ([^)]+)\).*/\1/p' \
                    | tee $out
                '';
              in
              rec {
                src = ./.;
                strictDeps = true;
                cargoVendorDir = craneLib.vendorMultipleCargoDeps {
                  inherit (craneLib.findCargoFiles src) cargoConfigs;
                  cargoLockList = [
                    ./Cargo.lock
                    "${rust}/lib/rustlib/src/rust/library/Cargo.lock"
                  ];
                };
                env.RUSTFLAGS = "-Ctarget-cpu=${builtins.readFile targetCpu}";
              }
            );
          cargoArtifacts = craneLib.buildDepsOnly craneCommonArgs;
        in
        {
          _module.args = { inherit inputs pkgs rust; };
          treefmt =
            { ... }:
            {
              projectRootFile = ".git/config";
              settings.global.excludes = [
                "bindings/mlkem-native-rs/mlkem-native/**"
                "bindings/pqmagic-rs/pqmagic/**"
              ];
              programs.nixfmt.enable = true;
              programs.taplo.enable = true;
              programs.rustfmt = {
                enable = true;
                package = rust;
              };
              programs.yamlfmt.enable = true;
            };
          devShells.default = pkgs.mkShell {
            inherit (pkgs) stdenv;
            inputsFrom = [ config.treefmt.build.devShell ];
            buildInputs =
              lib.singleton rust
              ++ (with pkgs; [
                llvmPackages.bolt
                cargo-show-asm
                cargo-flamegraph
                cargo-edit
                cargo-nextest
                cargo-llvm-cov
                cargo-criterion
                gnuplot
                pprof
                perf
                taplo
              ]);
          };
          checks = {
            clippy = craneLib.cargoClippy (
              craneCommonArgs
              // {
                inherit cargoArtifacts;
                cargoClippyExtraArgs = "--all-targets --all-features";
              }
            );
            test = craneLib.cargoNextest (craneCommonArgs // { inherit cargoArtifacts; });
          };
          packages = {
            benchmark = craneLib.mkCargoDerivation (
              craneCommonArgs
              // {
                inherit cargoArtifacts;
                pnameSuffix = "-benchmark";
                nativeBuildInputs = with pkgs; [
                  cargo-pgo
                  gnuplot
                  pprof
                ];
                buildPhaseCargoCommand = ''
                  cargo bench
                  cargo pgo bench -- -- --profile-time 10
                  cargo pgo optimize bench
                '';
                installPhaseCommand = "cp -r target/criterion $out";
              }
            );
            coverage = craneLib.mkCargoDerivation (
              craneCommonArgs
              // {
                inherit cargoArtifacts;
                pnameSuffix = "-coverage";
                nativeBuildInputs = with pkgs; [
                  cargo-nextest
                  cargo-llvm-cov
                ];
                buildPhaseCargoCommand = ''
                  cargo llvm-cov nextest --all-features
                '';
                installPhaseCommand = ''
                  cargo llvm-cov report --html --output-dir $out/html
                  cargo llvm-cov report --cobertura --output-path $out/coverage.xml
                  cargo llvm-cov report --lcov --output-path $out/coverage.lcov
                  cargo llvm-cov report --json --output-path $out/coverage.json
                '';
              }
            );
          };
        };
    };
}
