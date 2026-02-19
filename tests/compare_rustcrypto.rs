//! Byte-for-byte comparison against the RustCrypto `ml-kem` crate.
//! Both implement FIPS 203 — deterministic operations must produce identical
//! output.

use kem_rs::{MlKem512, MlKem768, MlKem1024, decapsulate, encapsulate_derand, keypair_derand};
use ml_kem::kem::Decapsulate;
use ml_kem::{EncapsulateDeterministic, EncodedSizeUser, KemCore};

fn keygen_coins(tag: u8) -> ([u8; 32], [u8; 32], [u8; 64]) {
    let full: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(37)));
    (
        full[..32].try_into().unwrap(),
        full[32..].try_into().unwrap(),
        full,
    )
}

fn enc_coins(tag: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(53)))
}

macro_rules! compare_tests {
    ($ours:ty, $theirs:ty, $mod:ident) => {
        mod $mod {
            use super::*;

            #[test]
            fn keygen_bytes_match() {
                for tag in 0..8u8 {
                    let (d, z, full) = keygen_coins(tag);
                    let (our_pk, our_sk) = keypair_derand::<$ours>(&full);
                    let (rc_dk, rc_ek) =
                        <$theirs>::generate_deterministic(&d.into(), &z.into());

                    let rc_pk = rc_ek.as_bytes();
                    let rc_sk = rc_dk.as_bytes();
                    assert_eq!(our_pk.as_ref(), &rc_pk[..], "ek mismatch (tag={tag})");
                    assert_eq!(our_sk.as_ref(), &rc_sk[..], "dk mismatch (tag={tag})");
                }
            }

            #[test]
            fn encaps_bytes_match() {
                let (d, z, full) = keygen_coins(0);
                let (our_pk, _) = keypair_derand::<$ours>(&full);
                let (_, rc_ek) =
                    <$theirs>::generate_deterministic(&d.into(), &z.into());

                for tag in 0..8u8 {
                    let m = enc_coins(tag);
                    let (our_ct, our_ss) = encapsulate_derand::<$ours>(&our_pk, &m);
                    let (rc_ct, rc_ss) =
                        rc_ek.encapsulate_deterministic(&m.into()).unwrap();

                    assert_eq!(our_ct.as_ref(), &rc_ct[..], "ct mismatch (tag={tag})");
                    assert_eq!(our_ss.as_ref(), &rc_ss[..], "ss mismatch (tag={tag})");
                }
            }

            #[test]
            fn decaps_bytes_match() {
                for tag in 0..8u8 {
                    let (d, z, full) = keygen_coins(tag);
                    let m = enc_coins(tag);

                    let (our_pk, our_sk) = keypair_derand::<$ours>(&full);
                    let (our_ct, our_ss_enc) = encapsulate_derand::<$ours>(&our_pk, &m);
                    let our_ss_dec = decapsulate::<$ours>(&our_ct, &our_sk);

                    let (rc_dk, rc_ek) =
                        <$theirs>::generate_deterministic(&d.into(), &z.into());
                    let (rc_ct, _) =
                        rc_ek.encapsulate_deterministic(&m.into()).unwrap();
                    let rc_ss_dec = rc_dk.decapsulate(&rc_ct).unwrap();

                    assert_eq!(our_ss_enc.as_ref(), our_ss_dec.as_ref());
                    assert_eq!(
                        our_ss_dec.as_ref(),
                        &rc_ss_dec[..],
                        "decaps ss mismatch (tag={tag})"
                    );
                }
            }
        }
    };
}

compare_tests!(MlKem512, ml_kem::MlKem512, mlkem512);
compare_tests!(MlKem768, ml_kem::MlKem768, mlkem768);
compare_tests!(MlKem1024, ml_kem::MlKem1024, mlkem1024);
