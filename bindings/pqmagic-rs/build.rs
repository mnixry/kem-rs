use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=pqmagic/kem/ml_kem/std");
    println!("cargo:rerun-if-changed=pqmagic/hash/keccak");
    println!("cargo:rerun-if-changed=pqmagic/utils");

    match env::var("CARGO_CFG_TARGET_FAMILY") {
        Ok(family) if family == "unix" => {}
        _ => return Ok(()),
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let pqmagic_dir = manifest_dir.join("pqmagic");
    let ml_kem_dir = pqmagic_dir.join("kem").join("ml_kem").join("std");

    let ml_kem_sources = [
        "kem.c",
        "indcpa.c",
        "ntt.c",
        "poly.c",
        "polyvec.c",
        "cbd.c",
        "reduce.c",
        "verify.c",
        "symmetric.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(&ml_kem_dir)
        .include(&pqmagic_dir)
        .include(pqmagic_dir.join("include"))
        .include(pqmagic_dir.join("utils"))
        .include(pqmagic_dir.join("hash").join("keccak"))
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("-mtune=native")
        .flag_if_supported("-fuse-ld=lld")
        .warnings(false);

    for mode in [512, 768, 1024] {
        for src in &ml_kem_sources {
            let stem = src.strip_suffix(".c").unwrap();
            let stub_name = format!("gen_pqmagic_{mode}_{stem}.c");
            let content = format!(
                "#define ML_KEM_MODE {mode}\n\
                 #define USE_SHAKE\n\
                 #include \"{src}\"\n"
            );
            std::fs::write(out_dir.join(&stub_name), content)?;
            build.file(out_dir.join(stub_name));
        }
    }

    build.file(pqmagic_dir.join("hash").join("keccak").join("fips202.c"));
    build.file(pqmagic_dir.join("utils").join("randombytes.c"));

    build.compile("pqmagic_ml_kem");

    Ok(())
}
