use std::collections::HashMap;

use kem_rs::{
    Ciphertext, MlKem512, MlKem768, MlKem1024, ParameterSet, PublicKey, SecretKey, decapsulate,
    encapsulate_derand,
    hash::hash_h,
    keypair_derand,
    params::{ByteArray, Q, SYMBYTES},
};
use serde::{Deserialize, de};

const KEYGEN_PROMPT_JSON: &[u8] = include_bytes!("data/acvp/ML-KEM-keyGen-FIPS203/prompt.json");
const KEYGEN_EXPECTED_JSON: &[u8] =
    include_bytes!("data/acvp/ML-KEM-keyGen-FIPS203/expectedResults.json");
const ENCAP_PROMPT_JSON: &[u8] = include_bytes!("data/acvp/ML-KEM-encapDecap-FIPS203/prompt.json");
const ENCAP_EXPECTED_JSON: &[u8] =
    include_bytes!("data/acvp/ML-KEM-encapDecap-FIPS203/expectedResults.json");

fn parse_json<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> T {
    serde_json::from_slice(bytes).expect("json")
}

fn de_hex_vec<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>, {
    let encoded = <&str>::deserialize(deserializer)?;
    hex::decode(encoded).map_err(de::Error::custom)
}

fn de_hex_32<'de, D>(deserializer: D) -> Result<[u8; SYMBYTES], D::Error>
where
    D: serde::Deserializer<'de>, {
    let bytes = de_hex_vec(deserializer)?;
    bytes
        .try_into()
        .map_err(|_| de::Error::custom("expected 32 bytes"))
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct HexBytes(#[serde(deserialize_with = "de_hex_vec")] Vec<u8>);

impl<'a> From<&'a HexBytes> for &'a [u8] {
    fn from(HexBytes(bytes): &'a HexBytes) -> Self {
        bytes
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct Hex32(#[serde(deserialize_with = "de_hex_32")] [u8; SYMBYTES]);

impl<'a> From<&'a Hex32> for &'a [u8; SYMBYTES] {
    fn from(Hex32(bytes): &'a Hex32) -> Self {
        bytes
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestVectorSet<G> {
    test_groups: Vec<G>,
}

#[derive(Clone, Copy, Deserialize)]
enum ParameterSetName {
    #[serde(rename = "ML-KEM-512")]
    MlKem512,
    #[serde(rename = "ML-KEM-768")]
    MlKem768,
    #[serde(rename = "ML-KEM-1024")]
    MlKem1024,
}

impl ParameterSetName {
    fn run<T>(
        self, f512: impl FnOnce() -> T, f768: impl FnOnce() -> T, f1024: impl FnOnce() -> T,
    ) -> T {
        match self {
            Self::MlKem512 => f512(),
            Self::MlKem768 => f768(),
            Self::MlKem1024 => f1024(),
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
enum EncapFunction {
    #[serde(rename = "encapsulation")]
    Encapsulation,
    #[serde(rename = "decapsulation")]
    Decapsulation,
    #[serde(rename = "decapsulationKeyCheck")]
    DecapsulationKeyCheck,
    #[serde(rename = "encapsulationKeyCheck")]
    EncapsulationKeyCheck,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeygenPromptGroup {
    tg_id: u64,
    parameter_set: ParameterSetName,
    tests: Vec<KeygenPromptTest>,
}

#[derive(Deserialize)]
struct KeygenPromptTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    d: Hex32,
    z: Hex32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeygenExpectedGroup {
    tg_id: u64,
    tests: Vec<KeygenExpectedTest>,
}

#[derive(Deserialize)]
struct KeygenExpectedTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    ek: HexBytes,
    dk: HexBytes,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncapPromptGroup {
    tg_id: u64,
    parameter_set: ParameterSetName,
    function: EncapFunction,
    tests: Vec<EncapPromptTest>,
}

#[derive(Deserialize)]
struct EncapPromptTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    #[serde(default)]
    ek: Option<HexBytes>,
    #[serde(default)]
    dk: Option<HexBytes>,
    #[serde(default)]
    c: Option<HexBytes>,
    #[serde(default)]
    m: Option<Hex32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncapExpectedGroup {
    tg_id: u64,
    tests: Vec<EncapExpectedTest>,
}

#[derive(Deserialize)]
struct EncapExpectedTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    #[serde(default)]
    c: Option<HexBytes>,
    #[serde(default)]
    k: Option<Hex32>,
    #[serde(default, rename = "testPassed")]
    test_passed: Option<bool>,
}

fn by_tc_id<T>(tests: &[T], id_fn: impl Fn(&T) -> u64) -> HashMap<u64, &T> {
    tests.iter().map(|test| (id_fn(test), test)).collect()
}

macro_rules! require_hex {
    ($struct:ident, $($field:ident), *) => {
        $(
            let $field = $struct.$field.as_ref().expect(stringify!($field)).into();
        )*
    };
}

fn to_byte_array<A: ByteArray>(bytes: &[u8], expected_len: usize) -> A {
    assert_eq!(bytes.len(), expected_len);
    let mut out = A::zeroed();
    out.as_mut().copy_from_slice(bytes);
    out
}

#[allow(clippy::similar_names)]
fn run_keygen_case<P: ParameterSet>(
    d: &[u8; SYMBYTES], z: &[u8; SYMBYTES], expected_ek: &[u8], expected_dk: &[u8],
) {
    let mut coins = [0u8; 2 * SYMBYTES];
    coins[..SYMBYTES].copy_from_slice(d);
    coins[SYMBYTES..].copy_from_slice(z);

    let (ek, dk) = keypair_derand::<P>(&coins);
    assert_eq!(ek.as_ref(), expected_ek);
    assert_eq!(dk.as_ref(), expected_dk);
}

fn run_encapsulation_case<P: ParameterSet>(
    ek: &[u8], m: &[u8; SYMBYTES], expected_c: &[u8], expected_k: &[u8; SYMBYTES],
) {
    let ek_arr = to_byte_array::<P::PkArray>(ek, P::PK_BYTES);
    let ek = PublicKey::<P>::from(&ek_arr);

    let (c, k) = encapsulate_derand::<P>(&ek, m);
    assert_eq!(c.as_ref(), expected_c);
    assert_eq!(k.as_ref(), expected_k);
}

fn run_decapsulation_case<P: ParameterSet>(dk: &[u8], c: &[u8], expected_k: &[u8; SYMBYTES]) {
    let dk_arr = to_byte_array::<P::SkArray>(dk, P::SK_BYTES);
    let c_arr = to_byte_array::<P::CtArray>(c, P::CT_BYTES);
    let dk = SecretKey::<P>::from(&dk_arr);
    let c = Ciphertext::<P>::from(&c_arr);
    let k = decapsulate::<P>(&c, &dk);
    assert_eq!(k.as_ref(), expected_k);
}

fn encapsulation_key_check<P: ParameterSet>(ek: &[u8]) -> bool {
    if ek.len() != P::PK_BYTES {
        return false;
    }

    let mut chunks = ek[..P::POLYVEC_BYTES].chunks_exact(3);
    for chunk in &mut chunks {
        let b0 = u16::from(chunk[0]);
        let b1 = u16::from(chunk[1]);
        let b2 = u16::from(chunk[2]);
        let c0 = b0 | ((b1 & 0x0F) << 8);
        let c1 = (b1 >> 4) | (b2 << 4);
        if c0 >= Q as u16 || c1 >= Q as u16 {
            return false;
        }
    }

    chunks.remainder().is_empty()
}

fn decapsulation_key_check<P: ParameterSet>(dk: &[u8]) -> bool {
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

#[test]
fn acvp_keygen_vectors() {
    let prompt: TestVectorSet<KeygenPromptGroup> = parse_json(KEYGEN_PROMPT_JSON);
    let expected: TestVectorSet<KeygenExpectedGroup> = parse_json(KEYGEN_EXPECTED_JSON);

    let expected_groups = expected
        .test_groups
        .iter()
        .map(|group| (group.tg_id, group))
        .collect::<HashMap<_, _>>();
    assert_eq!(prompt.test_groups.len(), expected_groups.len());

    for prompt_group in &prompt.test_groups {
        let expected_group = expected_groups.get(&prompt_group.tg_id).expect("tg");
        let expected_tests = by_tc_id(&expected_group.tests, |test| test.tc_id);
        assert_eq!(prompt_group.tests.len(), expected_tests.len());

        for prompt_test in &prompt_group.tests {
            let expected_test = expected_tests.get(&prompt_test.tc_id).expect("tc");
            prompt_group.parameter_set.run(
                || {
                    run_keygen_case::<MlKem512>(
                        (&prompt_test.d).into(),
                        (&prompt_test.z).into(),
                        (&expected_test.ek).into(),
                        (&expected_test.dk).into(),
                    );
                },
                || {
                    run_keygen_case::<MlKem768>(
                        (&prompt_test.d).into(),
                        (&prompt_test.z).into(),
                        (&expected_test.ek).into(),
                        (&expected_test.dk).into(),
                    );
                },
                || {
                    run_keygen_case::<MlKem1024>(
                        (&prompt_test.d).into(),
                        (&prompt_test.z).into(),
                        (&expected_test.ek).into(),
                        (&expected_test.dk).into(),
                    );
                },
            );
        }
    }
}

#[test]
fn acvp_encap_decap_vectors() {
    let prompt: TestVectorSet<EncapPromptGroup> = parse_json(ENCAP_PROMPT_JSON);
    let expected: TestVectorSet<EncapExpectedGroup> = parse_json(ENCAP_EXPECTED_JSON);

    let expected_groups = expected
        .test_groups
        .iter()
        .map(|group| (group.tg_id, group))
        .collect::<HashMap<_, _>>();
    assert_eq!(prompt.test_groups.len(), expected_groups.len());

    for prompt_group in &prompt.test_groups {
        let expected_group = expected_groups.get(&prompt_group.tg_id).expect("tg");
        let expected_tests = by_tc_id(&expected_group.tests, |test| test.tc_id);
        assert_eq!(prompt_group.tests.len(), expected_tests.len());

        for prompt_test in &prompt_group.tests {
            let expected_test = expected_tests.get(&prompt_test.tc_id).expect("tc");

            match prompt_group.function {
                EncapFunction::Encapsulation => {
                    require_hex!(prompt_test, ek, m);
                    require_hex!(expected_test, c, k);

                    prompt_group.parameter_set.run(
                        || run_encapsulation_case::<MlKem512>(ek, m, c, k),
                        || run_encapsulation_case::<MlKem768>(ek, m, c, k),
                        || run_encapsulation_case::<MlKem1024>(ek, m, c, k),
                    );
                }
                EncapFunction::Decapsulation => {
                    require_hex!(prompt_test, dk, c);
                    require_hex!(expected_test, k);

                    prompt_group.parameter_set.run(
                        || run_decapsulation_case::<MlKem512>(dk, c, k),
                        || run_decapsulation_case::<MlKem768>(dk, c, k),
                        || run_decapsulation_case::<MlKem1024>(dk, c, k),
                    );
                }
                EncapFunction::DecapsulationKeyCheck => {
                    require_hex!(prompt_test, dk);
                    let expected_passed = expected_test.test_passed.expect("testPassed");
                    let actual = prompt_group.parameter_set.run(
                        || decapsulation_key_check::<MlKem512>(dk),
                        || decapsulation_key_check::<MlKem768>(dk),
                        || decapsulation_key_check::<MlKem1024>(dk),
                    );
                    assert_eq!(actual, expected_passed);
                }
                EncapFunction::EncapsulationKeyCheck => {
                    require_hex!(prompt_test, ek);
                    let expected_passed = expected_test.test_passed.expect("testPassed");
                    let actual = prompt_group.parameter_set.run(
                        || encapsulation_key_check::<MlKem512>(ek),
                        || encapsulation_key_check::<MlKem768>(ek),
                        || encapsulation_key_check::<MlKem1024>(ek),
                    );
                    assert_eq!(actual, expected_passed);
                }
            }
        }
    }
}
