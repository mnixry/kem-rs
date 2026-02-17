use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use kem_rs::{
    Ciphertext, MlKem512, MlKem768, MlKem1024, MlKemParams, PublicKey, SecretKey, decapsulate,
    encapsulate_derand,
    hash::hash_h,
    keypair_derand,
    params::{ByteArray, Q, SYMBYTES},
};
use serde_json::Value;

const ACVP_JSON_ROOT: &str = "tests/data/acvp/gen-val/json-files";

fn read_json(path: &Path) -> Value {
    serde_json::from_reader(File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}")))
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn acvp_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(ACVP_JSON_ROOT)
        .join(relative)
}

fn get_u64(value: &Value, field: &str) -> u64 {
    value
        .get(field)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("missing/invalid u64 field `{field}`"))
}

fn get_str<'a>(value: &'a Value, field: &str) -> &'a str {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing/invalid string field `{field}`"))
}

fn get_bool(value: &Value, field: &str) -> bool {
    value
        .get(field)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("missing/invalid bool field `{field}`"))
}

fn get_hex(value: &Value, field: &str) -> Vec<u8> {
    let encoded = get_str(value, field);
    hex::decode(encoded).unwrap_or_else(|e| panic!("invalid hex for `{field}`: {e}"))
}

fn tests_by_tc_id(group: &Value) -> HashMap<u64, &Value> {
    group
        .get("tests")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("group missing `tests`: {group:?}"))
        .iter()
        .map(|test| (get_u64(test, "tcId"), test))
        .collect()
}

fn to_byte_array<A: ByteArray>(bytes: &[u8], expected_len: usize, field: &str) -> A {
    assert_eq!(
        bytes.len(),
        expected_len,
        "length mismatch for `{field}`: expected {expected_len}, got {}",
        bytes.len()
    );
    let mut out = A::zeroed();
    out.as_mut().copy_from_slice(bytes);
    out
}

fn run_keygen_case<P: MlKemParams>(d: &[u8], z: &[u8], expected_ek: &[u8], expected_dk: &[u8]) {
    assert_eq!(d.len(), SYMBYTES, "`d` must be 32 bytes");
    assert_eq!(z.len(), SYMBYTES, "`z` must be 32 bytes");

    let mut coins = [0u8; 2 * SYMBYTES];
    coins[..SYMBYTES].copy_from_slice(d);
    coins[SYMBYTES..].copy_from_slice(z);

    let (ek, dk) = keypair_derand::<P>(&coins);
    assert_eq!(ek.as_bytes(), expected_ek, "encapsulation key mismatch");
    assert_eq!(dk.as_bytes(), expected_dk, "decapsulation key mismatch");
}

fn run_encapsulation_case<P: MlKemParams>(
    ek: &[u8], m: &[u8], expected_c: &[u8], expected_k: &[u8],
) {
    assert_eq!(m.len(), SYMBYTES, "`m` must be 32 bytes");
    assert_eq!(expected_k.len(), SYMBYTES, "`k` must be 32 bytes");

    let mut m_arr = [0u8; SYMBYTES];
    m_arr.copy_from_slice(m);
    let ek_arr = to_byte_array::<P::PkArray>(ek, P::PK_BYTES, "ek");
    let ek = PublicKey::<P>::from_bytes(ek_arr);

    let (c, k) = encapsulate_derand::<P>(&ek, &m_arr);
    assert_eq!(c.as_bytes(), expected_c, "ciphertext mismatch");
    assert_eq!(k.as_bytes(), expected_k, "shared secret mismatch");
}

fn run_decapsulation_case<P: MlKemParams>(dk: &[u8], c: &[u8], expected_k: &[u8]) {
    assert_eq!(expected_k.len(), SYMBYTES, "`k` must be 32 bytes");

    let dk_arr = to_byte_array::<P::SkArray>(dk, P::SK_BYTES, "dk");
    let c_arr = to_byte_array::<P::CtArray>(c, P::CT_BYTES, "c");
    let dk = SecretKey::<P>::from_bytes(dk_arr);
    let c = Ciphertext::<P>::from_bytes(c_arr);
    let k = decapsulate::<P>(&c, &dk);

    assert_eq!(k.as_bytes(), expected_k, "shared secret mismatch");
}

fn encapsulation_key_check<P: MlKemParams>(ek: &[u8]) -> bool {
    if ek.len() != P::PK_BYTES {
        return false;
    }

    let mut chunks = ek[..P::POLYVEC_BYTES].chunks_exact(3);
    for chunk in &mut chunks {
        let b0 = chunk[0] as u16;
        let b1 = chunk[1] as u16;
        let b2 = chunk[2] as u16;
        let c0 = b0 | ((b1 & 0x0F) << 8);
        let c1 = (b1 >> 4) | (b2 << 4);
        if c0 >= Q as u16 || c1 >= Q as u16 {
            return false;
        }
    }

    chunks.remainder().is_empty()
}

fn decapsulation_key_check<P: MlKemParams>(dk: &[u8]) -> bool {
    if dk.len() != P::SK_BYTES {
        return false;
    }

    let ek_start = P::INDCPA_SK_BYTES;
    let ek_end = ek_start + P::PK_BYTES;
    let h_start = P::SK_BYTES - 2 * SYMBYTES;
    let h_end = h_start + SYMBYTES;

    let ek = &dk[ek_start..ek_end];
    let h = &dk[h_start..h_end];

    encapsulation_key_check::<P>(ek) && hash_h(ek).as_slice() == h
}

fn with_param_set<T>(
    parameter_set: &str, f512: impl FnOnce() -> T, f768: impl FnOnce() -> T,
    f1024: impl FnOnce() -> T,
) -> T {
    match parameter_set {
        "ML-KEM-512" => f512(),
        "ML-KEM-768" => f768(),
        "ML-KEM-1024" => f1024(),
        _ => panic!("unsupported parameter set `{parameter_set}`"),
    }
}

#[test]
fn acvp_keygen_vectors() {
    let prompt = read_json(&acvp_path("ML-KEM-keyGen-FIPS203/prompt.json"));
    let expected = read_json(&acvp_path("ML-KEM-keyGen-FIPS203/expectedResults.json"));

    let prompt_groups = prompt
        .get("testGroups")
        .and_then(Value::as_array)
        .expect("keygen prompt missing testGroups");
    let expected_groups = expected
        .get("testGroups")
        .and_then(Value::as_array)
        .expect("keygen expectedResults missing testGroups");
    let expected_by_tgid: HashMap<u64, &Value> = expected_groups
        .iter()
        .map(|g| (get_u64(g, "tgId"), g))
        .collect();

    assert_eq!(
        prompt_groups.len(),
        expected_by_tgid.len(),
        "keygen group count mismatch"
    );

    for prompt_group in prompt_groups {
        let tg_id = get_u64(prompt_group, "tgId");
        let parameter_set = get_str(prompt_group, "parameterSet");
        let prompt_tests = tests_by_tc_id(prompt_group);
        let expected_group = expected_by_tgid
            .get(&tg_id)
            .unwrap_or_else(|| panic!("missing expected keygen group tgId={tg_id}"));
        let expected_tests = tests_by_tc_id(expected_group);

        assert_eq!(
            prompt_tests.len(),
            expected_tests.len(),
            "keygen test count mismatch for tgId={tg_id}"
        );

        for (tc_id, prompt_test) in prompt_tests {
            let expected_test = expected_tests.get(&tc_id).unwrap_or_else(|| {
                panic!("missing expected keygen test tgId={tg_id} tcId={tc_id}")
            });

            let d = get_hex(prompt_test, "d");
            let z = get_hex(prompt_test, "z");
            let ek = get_hex(expected_test, "ek");
            let dk = get_hex(expected_test, "dk");

            with_param_set(
                parameter_set,
                || run_keygen_case::<MlKem512>(&d, &z, &ek, &dk),
                || run_keygen_case::<MlKem768>(&d, &z, &ek, &dk),
                || run_keygen_case::<MlKem1024>(&d, &z, &ek, &dk),
            );
        }
    }
}

#[test]
fn acvp_encap_decap_vectors() {
    let prompt = read_json(&acvp_path("ML-KEM-encapDecap-FIPS203/prompt.json"));
    let expected = read_json(&acvp_path("ML-KEM-encapDecap-FIPS203/expectedResults.json"));

    let prompt_groups = prompt
        .get("testGroups")
        .and_then(Value::as_array)
        .expect("encapDecap prompt missing testGroups");
    let expected_groups = expected
        .get("testGroups")
        .and_then(Value::as_array)
        .expect("encapDecap expectedResults missing testGroups");
    let expected_by_tgid: HashMap<u64, &Value> = expected_groups
        .iter()
        .map(|g| (get_u64(g, "tgId"), g))
        .collect();

    assert_eq!(
        prompt_groups.len(),
        expected_by_tgid.len(),
        "encapDecap group count mismatch"
    );

    for prompt_group in prompt_groups {
        let tg_id = get_u64(prompt_group, "tgId");
        let parameter_set = get_str(prompt_group, "parameterSet");
        let function = get_str(prompt_group, "function");
        let prompt_tests = tests_by_tc_id(prompt_group);
        let expected_group = expected_by_tgid
            .get(&tg_id)
            .unwrap_or_else(|| panic!("missing expected encapDecap group tgId={tg_id}"));
        let expected_tests = tests_by_tc_id(expected_group);

        assert_eq!(
            prompt_tests.len(),
            expected_tests.len(),
            "encapDecap test count mismatch for tgId={tg_id}"
        );

        for (tc_id, prompt_test) in prompt_tests {
            let expected_test = expected_tests.get(&tc_id).unwrap_or_else(|| {
                panic!("missing expected encapDecap test tgId={tg_id} tcId={tc_id}")
            });

            match function {
                "encapsulation" => {
                    let ek = get_hex(prompt_test, "ek");
                    let m = get_hex(prompt_test, "m");
                    let c = get_hex(expected_test, "c");
                    let k = get_hex(expected_test, "k");

                    with_param_set(
                        parameter_set,
                        || run_encapsulation_case::<MlKem512>(&ek, &m, &c, &k),
                        || run_encapsulation_case::<MlKem768>(&ek, &m, &c, &k),
                        || run_encapsulation_case::<MlKem1024>(&ek, &m, &c, &k),
                    );
                }
                "decapsulation" => {
                    let dk = get_hex(prompt_test, "dk");
                    let c = get_hex(prompt_test, "c");
                    let k = get_hex(expected_test, "k");

                    with_param_set(
                        parameter_set,
                        || run_decapsulation_case::<MlKem512>(&dk, &c, &k),
                        || run_decapsulation_case::<MlKem768>(&dk, &c, &k),
                        || run_decapsulation_case::<MlKem1024>(&dk, &c, &k),
                    );
                }
                "decapsulationKeyCheck" => {
                    let dk = get_hex(prompt_test, "dk");
                    let expected_passed = get_bool(expected_test, "testPassed");
                    let actual = with_param_set(
                        parameter_set,
                        || decapsulation_key_check::<MlKem512>(&dk),
                        || decapsulation_key_check::<MlKem768>(&dk),
                        || decapsulation_key_check::<MlKem1024>(&dk),
                    );
                    assert_eq!(
                        actual, expected_passed,
                        "decapsulation key check mismatch for tgId={tg_id} tcId={tc_id}"
                    );
                }
                "encapsulationKeyCheck" => {
                    let ek = get_hex(prompt_test, "ek");
                    let expected_passed = get_bool(expected_test, "testPassed");
                    let actual = with_param_set(
                        parameter_set,
                        || encapsulation_key_check::<MlKem512>(&ek),
                        || encapsulation_key_check::<MlKem768>(&ek),
                        || encapsulation_key_check::<MlKem1024>(&ek),
                    );
                    assert_eq!(
                        actual, expected_passed,
                        "encapsulation key check mismatch for tgId={tg_id} tcId={tc_id}"
                    );
                }
                _ => panic!("unsupported function `{function}` in tgId={tg_id}"),
            }
        }
    }
}
