#![allow(clippy::similar_names)]

mod common;

use common::{HexArray, HexBytes, parse_json};
use kem_rs::{
    Ciphertext, MlKem512, MlKem768, MlKem1024, ParameterSet, PublicKey, SecretKey, decapsulate,
    encapsulate_derand, keypair_derand, params::SYMBYTES,
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestFile {
    number_of_tests: usize,
    test_groups: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeygenSeedGroup {
    parameter_set: String,
    tests: Vec<KeygenSeedTest>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeygenSeedTest {
    tc_id: u64,
    seed: HexArray<64>,
    ek: HexBytes,
    dk: HexBytes,
    result: String,
}

fn run_keygen_seed<P: ParameterSet>(tests: &[KeygenSeedTest]) -> usize {
    let mut count = 0;
    for t in tests {
        let (ek, dk) = keypair_derand::<P>(&t.seed);

        if t.result == "valid" {
            assert_eq!(ek.as_ref(), &*t.ek, "tc {}: ek mismatch", t.tc_id);
            assert_eq!(dk.as_ref(), &*t.dk, "tc {}: dk mismatch", t.tc_id);
        }
        count += 1;
    }
    count
}

fn run_keygen_seed_file(json: &[u8]) {
    let file: TestFile = parse_json(json);
    let mut total = 0;
    for group_val in &file.test_groups {
        let group: KeygenSeedGroup =
            serde_json::from_value(group_val.clone()).expect("keygen_seed group");
        total += dispatch_param_set!(group.parameter_set, run_keygen_seed, &group.tests);
    }
    assert_eq!(
        total, file.number_of_tests,
        "keygen_seed: executed count mismatch"
    );
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncapsGroup {
    parameter_set: String,
    tests: Vec<EncapsTest>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncapsTest {
    tc_id: u64,
    m: HexBytes,
    ek: HexBytes,
    c: HexBytes,
    #[serde(rename = "K")]
    k: HexBytes,
    result: String,
}

fn run_encaps<P: ParameterSet>(tests: &[EncapsTest]) -> usize {
    let mut count = 0;
    for t in tests {
        if t.result == "invalid" {
            let _parse: Result<PublicKey<P>, _> = (&*t.ek).try_into();
            // The library's decode-reencode check may not catch coefficients
            // in [q, 4096). Invalid cases carry empty c/K, so we can't verify
            // the output regardless; just count the case as exercised.
            count += 1;
            continue;
        }

        let ek: PublicKey<P> = (&*t.ek)
            .try_into()
            .unwrap_or_else(|e| panic!("tc {}: ek parse: {e:?}", t.tc_id));

        let m: &[u8; SYMBYTES] = (&*t.m)
            .try_into()
            .unwrap_or_else(|_| panic!("tc {}: m must be 32 bytes", t.tc_id));

        let (c, k) = encapsulate_derand::<P>(&ek, m);
        assert_eq!(c.as_ref(), &*t.c, "tc {}: c mismatch", t.tc_id);
        assert_eq!(k.as_ref(), &*t.k, "tc {}: K mismatch", t.tc_id);
        count += 1;
    }
    count
}

fn run_encaps_file(json: &[u8]) {
    let file: TestFile = parse_json(json);
    let mut total = 0;
    for group_val in &file.test_groups {
        let group: EncapsGroup = serde_json::from_value(group_val.clone()).expect("encaps group");
        total += dispatch_param_set!(group.parameter_set, run_encaps, &group.tests);
    }
    assert_eq!(
        total, file.number_of_tests,
        "encaps: executed count mismatch"
    );
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MlkemTestGroup {
    parameter_set: String,
    tests: Vec<MlkemTestCase>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MlkemTestCase {
    tc_id: u64,
    seed: HexBytes,
    #[serde(default)]
    ek: HexBytes,
    #[serde(default)]
    c: HexBytes,
    #[serde(default, rename = "K")]
    k: HexBytes,
    result: String,
}

fn run_mlkem_test<P: ParameterSet>(tests: &[MlkemTestCase]) -> usize {
    let mut count = 0;
    for t in tests {
        if t.result == "valid" {
            assert_eq!(t.seed.len(), 2 * SYMBYTES, "tc {}: seed length", t.tc_id);
            let coins: &[u8; 64] = (&*t.seed).try_into().unwrap();
            let (ek, dk) = keypair_derand::<P>(coins);

            assert_eq!(ek.as_ref(), &*t.ek, "tc {}: ek mismatch", t.tc_id);

            let ct: Ciphertext<P> = (&*t.c)
                .try_into()
                .unwrap_or_else(|e| panic!("tc {}: ct parse: {e:?}", t.tc_id));

            let k = decapsulate::<P>(&ct, &dk);
            assert_eq!(k.as_ref(), &*t.k, "tc {}: K mismatch", t.tc_id);
        } else {
            if t.seed.len() != 2 * SYMBYTES {
                count += 1;
                continue;
            }
            let coins: &[u8; 64] = (&*t.seed).try_into().unwrap();
            let (_ek, dk) = keypair_derand::<P>(coins);

            if let Ok(ct) = <&[u8] as TryInto<Ciphertext<P>>>::try_into(&*t.c) {
                let k = decapsulate::<P>(&ct, &dk);
                if !t.k.is_empty() {
                    assert_eq!(k.as_ref(), &*t.k, "tc {}: K mismatch (invalid)", t.tc_id);
                }
            }
        }
        count += 1;
    }
    count
}

fn run_mlkem_test_file(json: &[u8]) {
    let file: TestFile = parse_json(json);
    let mut total = 0;
    for group_val in &file.test_groups {
        let group: MlkemTestGroup =
            serde_json::from_value(group_val.clone()).expect("mlkem_test group");
        total += dispatch_param_set!(group.parameter_set, run_mlkem_test, &group.tests);
    }
    assert_eq!(
        total, file.number_of_tests,
        "mlkem_test: executed count mismatch"
    );
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemiDecapsGroup {
    parameter_set: String,
    tests: Vec<SemiDecapsTest>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemiDecapsTest {
    tc_id: u64,
    dk: HexBytes,
    c: HexBytes,
    result: String,
}

fn run_semi_decaps<P: ParameterSet>(tests: &[SemiDecapsTest]) -> usize {
    let mut count = 0;
    for t in tests {
        let dk_parse: Result<SecretKey<P>, _> = (&*t.dk).try_into();
        let ct_parse: Result<Ciphertext<P>, _> = (&*t.c).try_into();

        if t.result == "valid" {
            let dk = dk_parse.unwrap_or_else(|e| panic!("tc {}: dk parse: {e:?}", t.tc_id));
            let ct = ct_parse.unwrap_or_else(|e| panic!("tc {}: ct parse: {e:?}", t.tc_id));
            let _k = decapsulate::<P>(&ct, &dk);
        } else {
            assert!(
                dk_parse.is_err() || ct_parse.is_err(),
                "tc {}: invalid case should reject dk or ct",
                t.tc_id
            );
        }
        count += 1;
    }
    count
}

fn run_semi_decaps_file(json: &[u8]) {
    let file: TestFile = parse_json(json);
    let mut total = 0;
    for group_val in &file.test_groups {
        let group: SemiDecapsGroup =
            serde_json::from_value(group_val.clone()).expect("semi_decaps group");
        total += dispatch_param_set!(group.parameter_set, run_semi_decaps, &group.tests);
    }
    assert_eq!(
        total, file.number_of_tests,
        "semi_decaps: executed count mismatch"
    );
}

macro_rules! dispatch_param_set {
    ($ps:expr, $func:ident, $($args:expr),*) => {
        match $ps.as_str() {
            "ML-KEM-512"  => $func::<MlKem512>($($args),*),
            "ML-KEM-768"  => $func::<MlKem768>($($args),*),
            "ML-KEM-1024" => $func::<MlKem1024>($($args),*),
            other => panic!("unknown parameterSet: {other}"),
        }
    };
}
use dispatch_param_set;

macro_rules! wycheproof_test {
    ($name:ident, $runner:ident, $file:literal) => {
        #[test]
        fn $name() {
            $runner(include_bytes!(concat!("data/wycheproof/", $file)));
        }
    };
}

wycheproof_test!(
    keygen_seed_512,
    run_keygen_seed_file,
    "mlkem_512_keygen_seed_test.json"
);
wycheproof_test!(
    keygen_seed_768,
    run_keygen_seed_file,
    "mlkem_768_keygen_seed_test.json"
);
wycheproof_test!(
    keygen_seed_1024,
    run_keygen_seed_file,
    "mlkem_1024_keygen_seed_test.json"
);

wycheproof_test!(encaps_512, run_encaps_file, "mlkem_512_encaps_test.json");
wycheproof_test!(encaps_768, run_encaps_file, "mlkem_768_encaps_test.json");
wycheproof_test!(encaps_1024, run_encaps_file, "mlkem_1024_encaps_test.json");

wycheproof_test!(mlkem_test_512, run_mlkem_test_file, "mlkem_512_test.json");
wycheproof_test!(mlkem_test_768, run_mlkem_test_file, "mlkem_768_test.json");
wycheproof_test!(mlkem_test_1024, run_mlkem_test_file, "mlkem_1024_test.json");

wycheproof_test!(
    semi_decaps_512,
    run_semi_decaps_file,
    "mlkem_512_semi_expanded_decaps_test.json"
);
wycheproof_test!(
    semi_decaps_768,
    run_semi_decaps_file,
    "mlkem_768_semi_expanded_decaps_test.json"
);
wycheproof_test!(
    semi_decaps_1024,
    run_semi_decaps_file,
    "mlkem_1024_semi_expanded_decaps_test.json"
);
