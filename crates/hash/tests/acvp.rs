use std::collections::HashMap;

use kem_hash::{hash_g, hash_h, shake128, shake256};
use serde::{Deserialize, de};

const SHA3_256_PROMPT: &[u8] = include_bytes!("data/acvp/SHA3-256/prompt.json");
const SHA3_256_EXPECTED: &[u8] = include_bytes!("data/acvp/SHA3-256/expectedResults.json");
const SHA3_512_PROMPT: &[u8] = include_bytes!("data/acvp/SHA3-512/prompt.json");
const SHA3_512_EXPECTED: &[u8] = include_bytes!("data/acvp/SHA3-512/expectedResults.json");
const SHAKE128_PROMPT: &[u8] = include_bytes!("data/acvp/SHAKE-128/prompt.json");
const SHAKE128_EXPECTED: &[u8] = include_bytes!("data/acvp/SHAKE-128/expectedResults.json");
const SHAKE256_PROMPT: &[u8] = include_bytes!("data/acvp/SHAKE-256/prompt.json");
const SHAKE256_EXPECTED: &[u8] = include_bytes!("data/acvp/SHAKE-256/expectedResults.json");

fn parse_json<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> T {
    serde_json::from_slice(bytes).expect("json")
}

fn de_hex<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>, {
    let encoded = <&str>::deserialize(deserializer)?;
    if encoded.is_empty() {
        return Ok(Vec::new());
    }
    hex::decode(encoded).map_err(de::Error::custom)
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct HexBytes(#[serde(deserialize_with = "de_hex")] Vec<u8>);

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VectorSet<G> {
    test_groups: Vec<G>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sha3PromptGroup {
    tg_id: u64,
    test_type: String,
    tests: Vec<Sha3PromptTest>,
}

#[derive(Deserialize)]
struct Sha3PromptTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    #[serde(default)]
    msg: Option<HexBytes>,
    #[serde(default)]
    len: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sha3ExpectedGroup {
    tg_id: u64,
    tests: Vec<Sha3ExpectedTest>,
}

#[derive(Deserialize)]
struct Sha3ExpectedTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    #[serde(default)]
    md: Option<HexBytes>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShakePromptGroup {
    tg_id: u64,
    test_type: String,
    tests: Vec<ShakePromptTest>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShakePromptTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    msg: HexBytes,
    len: usize,
    out_len: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShakeExpectedGroup {
    tg_id: u64,
    tests: Vec<ShakeExpectedTest>,
}

#[derive(Deserialize)]
struct ShakeExpectedTest {
    #[serde(rename = "tcId")]
    tc_id: u64,
    md: HexBytes,
}

fn by_tc_id<T>(tests: &[T], id_fn: impl Fn(&T) -> u64) -> HashMap<u64, &T> {
    tests.iter().map(|test| (id_fn(test), test)).collect()
}

fn run_sha3_aft(
    prompt_bytes: &[u8], expected_bytes: &[u8], hash_fn: impl Fn(&[u8]) -> Vec<u8>, name: &str,
) {
    let prompt: VectorSet<Sha3PromptGroup> = parse_json(prompt_bytes);
    let expected: VectorSet<Sha3ExpectedGroup> = parse_json(expected_bytes);

    let expected_groups: HashMap<_, _> =
        expected.test_groups.iter().map(|g| (g.tg_id, g)).collect();

    let mut tested = 0;
    for pg in &prompt.test_groups {
        if pg.test_type != "AFT" {
            continue;
        }
        let eg = expected_groups[&pg.tg_id];
        let expected_tests = by_tc_id(&eg.tests, |t| t.tc_id);

        for pt in &pg.tests {
            let Some(len) = pt.len else { continue };
            if len % 8 != 0 {
                continue;
            }
            let msg = match &pt.msg {
                Some(m) => &m.0,
                None => continue,
            };
            let msg_bytes = len / 8;
            assert!(msg.len() >= msg_bytes);

            let et = expected_tests[&pt.tc_id];
            let expected_md = et.md.as_ref().expect("md");
            let actual = hash_fn(&msg[..msg_bytes]);
            assert_eq!(actual, expected_md.0, "{name} mismatch tcId={}", pt.tc_id);
            tested += 1;
        }
    }
    assert!(tested > 0, "no {name} AFT tests ran");
    eprintln!("{name}: {tested} AFT vectors passed");
}

fn run_shake_aft(
    prompt_bytes: &[u8], expected_bytes: &[u8], shake_fn: impl Fn(&[u8], &mut [u8]), name: &str,
) {
    let prompt: VectorSet<ShakePromptGroup> = parse_json(prompt_bytes);
    let expected: VectorSet<ShakeExpectedGroup> = parse_json(expected_bytes);

    let expected_groups: HashMap<_, _> =
        expected.test_groups.iter().map(|g| (g.tg_id, g)).collect();

    let mut tested = 0;
    for pg in &prompt.test_groups {
        if pg.test_type != "AFT" {
            continue;
        }
        let eg = expected_groups[&pg.tg_id];
        let expected_tests = by_tc_id(&eg.tests, |t| t.tc_id);

        for pt in &pg.tests {
            if pt.len % 8 != 0 {
                continue;
            }
            let msg_bytes = pt.len / 8;
            let out_bytes = pt.out_len.div_ceil(8);
            let tail_bits = pt.out_len % 8;

            let et = expected_tests[&pt.tc_id];
            let mut actual = vec![0u8; out_bytes];
            shake_fn(&pt.msg.0[..msg_bytes], &mut actual);

            // ACVP places the meaningful tail bits in MSB positions;
            // Keccak squeezes them into LSB positions — shift to match.
            if tail_bits != 0
                && let Some(last) = actual.last_mut()
            {
                *last = last.wrapping_shl((8 - tail_bits) as u32);
            }

            assert_eq!(actual, et.md.0, "{name} mismatch tcId={}", pt.tc_id);
            tested += 1;
        }
    }
    assert!(tested > 0, "no {name} AFT tests ran");
    eprintln!("{name}: {tested} AFT vectors passed");
}

#[test]
fn acvp_sha3_256() {
    run_sha3_aft(
        SHA3_256_PROMPT,
        SHA3_256_EXPECTED,
        |m| hash_h(m).to_vec(),
        "SHA3-256",
    );
}

#[test]
fn acvp_sha3_512() {
    run_sha3_aft(
        SHA3_512_PROMPT,
        SHA3_512_EXPECTED,
        |m| hash_g(m).to_vec(),
        "SHA3-512",
    );
}

#[test]
fn acvp_shake128() {
    run_shake_aft(
        SHAKE128_PROMPT,
        SHAKE128_EXPECTED,
        |m, o| shake128(m, o),
        "SHAKE-128",
    );
}

#[test]
fn acvp_shake256() {
    run_shake_aft(
        SHAKE256_PROMPT,
        SHAKE256_EXPECTED,
        |m, o| shake256(m, o),
        "SHAKE-256",
    );
}
