use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=mlkem-native/mlkem");

    match env::var("CARGO_CFG_TARGET_FAMILY") {
        Ok(family) if family == "unix" => {}
        _ => return Ok(()),
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let mlkem_dir = manifest_dir.join("mlkem-native").join("mlkem");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    if !mlkem_dir.exists() {
        return Err("mlkem_dir does not exist".into());
    }

    let levels = vec![
        (512, "#define MLK_CONFIG_MULTILEVEL_WITH_SHARED 1"),
        (768, "#define MLK_CONFIG_MULTILEVEL_NO_SHARED"),
        (1024, "#define MLK_CONFIG_MULTILEVEL_NO_SHARED"),
    ];

    for (level, multilevel_define) in levels {
        let content = format!(
            "{multilevel_define}\n\
             #define MLK_CONFIG_PARAMETER_SET {level}\n\
             #include \"mlkem_native.c\"\n"
        );
        std::fs::write(out_dir.join(format!("gen_mlkem_{level}.c")), content)?;
    }

    std::fs::write(
        out_dir.join("gen_mlkem_asm.S"),
        "#define MLK_CONFIG_MULTILEVEL_WITH_SHARED 1\n\
         #include \"mlkem_native_asm.S\"\n",
    )?;

    let mut build = cc::Build::new();
    build
        .include(&mlkem_dir)
        .file(out_dir.join("gen_mlkem_512.c"))
        .file(out_dir.join("gen_mlkem_768.c"))
        .file(out_dir.join("gen_mlkem_1024.c"))
        .file(out_dir.join("gen_mlkem_asm.S"))
        .define("MLK_CONFIG_NAMESPACE_PREFIX", "PQCP_MLKEM_NATIVE_MLKEM")
        .define("MLK_CONFIG_USE_NATIVE_BACKEND_ARITH", None)
        .define("MLK_CONFIG_USE_NATIVE_BACKEND_FIPS202", None)
        .define("MLK_CONFIG_NO_RANDOMIZED_API", None)
        .opt_level(3)
        .flag_if_supported("-fuse-ld=lld")
        .warnings(false);

    match target_arch.as_str() {
        "x86_64" => {
            build.define("MLK_FORCE_X86_64", None);
            build.flag_if_supported("-mavx2");
            build.flag_if_supported("-mbmi2");
        }
        "aarch64" => {
            build.define("MLK_FORCE_AARCH64", None);
        }
        "riscv64" => {
            build.define("MLK_FORCE_RISCV64", None);
            build.flag_if_supported("-march=rv64gcv");
        }
        _ => {}
    }

    build.compile("mlkem_native");

    Ok(())
}
