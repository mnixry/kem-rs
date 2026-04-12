//! Byte-for-byte comparison against `PQMagic`'s ML-KEM (C reference, SHAKE
//! mode). Both implement FIPS 203 -- deterministic operations must produce
//! identical output.

use kem_rs::{MlKem512, MlKem768, MlKem1024, decapsulate, encapsulate_derand, keypair_derand};

fn keygen_coins(tag: u8) -> [u8; 64] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(37)))
}

fn enc_coins(tag: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(53)))
}

macro_rules! compare_tests {
    ($ours:ty, $pqmagic_mod:path, $mod:ident) => {
        mod $mod {
            use $pqmagic_mod as pqm;

            use super::*;

            #[test]
            fn keygen_bytes_match() {
                for tag in 0..8u8 {
                    let coins = keygen_coins(tag);
                    let (our_pk, our_sk) = keypair_derand::<$ours>(&coins);
                    let (pqm_pk, pqm_sk) = pqm::keypair_derand(&coins).unwrap();

                    assert_eq!(our_pk.as_ref(), &pqm_pk[..], "pk mismatch (tag={tag})");
                    assert_eq!(our_sk.as_ref(), &pqm_sk[..], "sk mismatch (tag={tag})");
                }
            }

            #[test]
            fn encaps_bytes_match() {
                let coins = keygen_coins(0);
                let (our_pk, _) = keypair_derand::<$ours>(&coins);
                let (pqm_pk, _) = pqm::keypair_derand(&coins).unwrap();

                for tag in 0..8u8 {
                    let m = enc_coins(tag);
                    let (our_ct, our_ss) = encapsulate_derand::<$ours>(&our_pk, &m);
                    let (pqm_ct, pqm_ss) = pqm::enc_derand(&pqm_pk, &m).unwrap();

                    assert_eq!(our_ct.as_ref(), &pqm_ct[..], "ct mismatch (tag={tag})");
                    assert_eq!(our_ss.as_ref(), &pqm_ss[..], "ss mismatch (tag={tag})");
                }
            }

            #[test]
            fn decaps_bytes_match() {
                for tag in 0..8u8 {
                    let coins = keygen_coins(tag);
                    let m = enc_coins(tag);

                    let (our_pk, our_sk) = keypair_derand::<$ours>(&coins);
                    let (our_ct, our_ss_enc) = encapsulate_derand::<$ours>(&our_pk, &m);
                    let our_ss_dec = decapsulate::<$ours>(&our_ct, &our_sk);

                    let (pqm_pk, pqm_sk) = pqm::keypair_derand(&coins).unwrap();
                    let (pqm_ct, _) = pqm::enc_derand(&pqm_pk, &m).unwrap();
                    let pqm_ss_dec = pqm::dec(&pqm_ct, &pqm_sk).unwrap();

                    assert_eq!(our_ss_enc.as_ref(), our_ss_dec.as_ref());
                    assert_eq!(
                        our_ss_dec.as_ref(),
                        &pqm_ss_dec[..],
                        "decaps ss mismatch (tag={tag})"
                    );
                }
            }
        }
    };
}

compare_tests!(MlKem512, pqmagic_rs::mlkem512, mlkem512);
compare_tests!(MlKem768, pqmagic_rs::mlkem768, mlkem768);
compare_tests!(MlKem1024, pqmagic_rs::mlkem1024, mlkem1024);
