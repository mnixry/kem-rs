//! NIST KAT (Known-Answer Test) verification.
//!
//! Reproduces the first KAT transcript for each ML-KEM parameter set using
//! the NIST AES-256-CTR DRBG, then verifies the SHA-256 hash of the formatted
//! output matches the expected values from PQClean META.yml.

use aes::cipher::{BlockEncrypt, KeyInit};
use sha2::{Digest, Sha256};

use kem_rs::{
    MlKem512, MlKem768, MlKem1024, MlKemParams,
    keypair_derand, encapsulate_derand, decapsulate,
};

// ---------------------------------------------------------------------------
// AES-256-CTR DRBG (NIST SP 800-90A, for KAT reproducibility only)
// ---------------------------------------------------------------------------

struct NistDrbg {
    key: [u8; 32],
    v: [u8; 16],
}

impl NistDrbg {
    fn new(entropy: &[u8; 48], personalization: Option<&[u8; 48]>) -> Self {
        let mut seed_material = [0u8; 48];
        seed_material.copy_from_slice(entropy);
        if let Some(ps) = personalization {
            for i in 0..48 {
                seed_material[i] ^= ps[i];
            }
        }
        let mut key = [0u8; 32];
        let mut v = [0u8; 16];
        Self::update(Some(&seed_material), &mut key, &mut v);
        NistDrbg { key, v }
    }

    fn update(provided_data: Option<&[u8; 48]>, key: &mut [u8; 32], v: &mut [u8; 16]) {
        let mut temp = [0u8; 48];
        for i in 0..3 {
            // Increment V (big-endian)
            for j in (0..16).rev() {
                if v[j] == 0xff {
                    v[j] = 0x00;
                } else {
                    v[j] += 1;
                    break;
                }
            }
            let cipher = aes::Aes256::new(key.as_slice().into());
            let mut block = aes::Block::clone_from_slice(v.as_slice());
            cipher.encrypt_block(&mut block);
            temp[16 * i..16 * (i + 1)].copy_from_slice(&block);
        }
        if let Some(data) = provided_data {
            for i in 0..48 {
                temp[i] ^= data[i];
            }
        }
        key.copy_from_slice(&temp[..32]);
        v.copy_from_slice(&temp[32..48]);
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut remaining = buf.len();
        let mut offset = 0;
        while remaining > 0 {
            // Increment V (big-endian)
            for j in (0..16).rev() {
                if self.v[j] == 0xff {
                    self.v[j] = 0x00;
                } else {
                    self.v[j] += 1;
                    break;
                }
            }
            let cipher = aes::Aes256::new(self.key.as_slice().into());
            let mut block = aes::Block::clone_from_slice(self.v.as_slice());
            cipher.encrypt_block(&mut block);
            if remaining > 15 {
                buf[offset..offset + 16].copy_from_slice(&block);
                offset += 16;
                remaining -= 16;
            } else {
                buf[offset..offset + remaining].copy_from_slice(&block[..remaining]);
                remaining = 0;
            }
        }
        Self::update(None, &mut self.key, &mut self.v);
    }
}

// ---------------------------------------------------------------------------
// KAT transcript helper
// ---------------------------------------------------------------------------

/// Format a byte slice as uppercase hex, matching PQClean's `fprintBstr`.
fn hex_upper(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "00".to_string();
    }
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

/// Run the first NIST KAT case for a given parameter set and return the
/// SHA-256 hash of the formatted transcript.
fn run_nist_kat_case<P: MlKemParams>() -> String {
    // Step 1: Init DRBG with entropy_input[i] = i
    let entropy: [u8; 48] = core::array::from_fn(|i| i as u8);
    let mut drbg = NistDrbg::new(&entropy, None);

    // Step 2: Draw 48-byte seed
    let mut seed = [0u8; 48];
    drbg.fill_bytes(&mut seed);

    // Step 3: Re-init DRBG with seed
    drbg = NistDrbg::new(&seed, None);

    // Step 4: Draw keypair coins (64 bytes = 2 * SYMBYTES)
    let mut keypair_coins = [0u8; 64];
    drbg.fill_bytes(&mut keypair_coins);

    // Step 5: Generate keypair
    let (pk, sk) = keypair_derand::<P>(&keypair_coins);

    // Step 6: Draw encapsulation coins (32 bytes = SYMBYTES)
    let mut enc_coins = [0u8; 32];
    drbg.fill_bytes(&mut enc_coins);

    // Step 7: Encapsulate
    let (ct, ss_enc) = encapsulate_derand::<P>(&pk, &enc_coins);

    // Step 8: Decapsulate and verify
    let ss_dec = decapsulate::<P>(&ct, &sk);
    assert_eq!(
        ss_enc.as_bytes(),
        ss_dec.as_bytes(),
        "KAT: encapsulate and decapsulate shared secrets must match"
    );

    // Step 9: Format transcript exactly like PQClean's nistkat.c
    let mut transcript = String::new();
    transcript.push_str("count = 0\n");
    transcript.push_str(&format!("seed = {}\n", hex_upper(&seed)));
    transcript.push_str(&format!("pk = {}\n", hex_upper(pk.as_bytes())));
    transcript.push_str(&format!("sk = {}\n", hex_upper(sk.as_bytes())));
    transcript.push_str(&format!("ct = {}\n", hex_upper(ct.as_bytes())));
    transcript.push_str(&format!("ss = {}\n", hex_upper(ss_enc.as_bytes())));

    // Step 10: SHA-256 of the transcript
    let mut hasher = Sha256::new();
    hasher.update(transcript.as_bytes());
    let hash = hasher.finalize();
    hex::encode(hash)
}

// ---------------------------------------------------------------------------
// NIST KAT SHA-256 checks (expected values from PQClean META.yml)
// ---------------------------------------------------------------------------

#[test]
fn nist_kat_mlkem512() {
    let hash = run_nist_kat_case::<MlKem512>();
    assert_eq!(
        hash,
        "c70041a761e01cd6426fa60e9fd6a4412c2be817386c8d0f3334898082512782",
        "ML-KEM-512 NIST KAT SHA-256 mismatch"
    );
}

#[test]
fn nist_kat_mlkem768() {
    let hash = run_nist_kat_case::<MlKem768>();
    assert_eq!(
        hash,
        "5352539586b6c3df58be6158a6250aeff402bd73060b0a3de68850ac074c17c3",
        "ML-KEM-768 NIST KAT SHA-256 mismatch"
    );
}

#[test]
fn nist_kat_mlkem1024() {
    let hash = run_nist_kat_case::<MlKem1024>();
    assert_eq!(
        hash,
        "f580d851e5fb27e6876e5e203fa18be4cdbfd49e05d48fec3d3992c8f43a13e6",
        "ML-KEM-1024 NIST KAT SHA-256 mismatch"
    );
}
