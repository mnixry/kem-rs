//! Byte-for-byte comparison against `mlkem-native` (C/asm reference).
//! Both implement FIPS 203 -- deterministic operations must produce identical
//! output.

use kem_rs::{MlKem512, MlKem768, MlKem1024, decapsulate, encapsulate_derand, keypair_derand};

fn keygen_coins(tag: u8) -> [u8; 64] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(37)))
}

fn enc_coins(tag: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(53)))
}

macro_rules! compare_tests {
    ($ours:ty, $native_mod:path, $mod:ident) => {
        mod $mod {
            use $native_mod as native;

            use super::*;

            #[test]
            fn keygen_bytes_match() {
                for tag in 0..8u8 {
                    let coins = keygen_coins(tag);
                    let (our_pk, our_sk) = keypair_derand::<$ours>(&coins);
                    let (nat_pk, nat_sk) = native::keypair_derand(&coins).unwrap();

                    assert_eq!(our_pk.as_ref(), &nat_pk[..], "pk mismatch (tag={tag})");
                    assert_eq!(our_sk.as_ref(), &nat_sk[..], "sk mismatch (tag={tag})");
                }
            }

            #[test]
            fn encaps_bytes_match() {
                let coins = keygen_coins(0);
                let (our_pk, _) = keypair_derand::<$ours>(&coins);
                let (nat_pk, _) = native::keypair_derand(&coins).unwrap();

                for tag in 0..8u8 {
                    let m = enc_coins(tag);
                    let (our_ct, our_ss) = encapsulate_derand::<$ours>(&our_pk, &m);
                    let (nat_ct, nat_ss) = native::enc_derand(&nat_pk, &m).unwrap();

                    assert_eq!(our_ct.as_ref(), &nat_ct[..], "ct mismatch (tag={tag})");
                    assert_eq!(our_ss.as_ref(), &nat_ss[..], "ss mismatch (tag={tag})");
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

                    let (nat_pk, nat_sk) = native::keypair_derand(&coins).unwrap();
                    let (nat_ct, _) = native::enc_derand(&nat_pk, &m).unwrap();
                    let nat_ss_dec = native::dec(&nat_ct, &nat_sk).unwrap();

                    assert_eq!(our_ss_enc.as_ref(), our_ss_dec.as_ref());
                    assert_eq!(
                        our_ss_dec.as_ref(),
                        &nat_ss_dec[..],
                        "decaps ss mismatch (tag={tag})"
                    );
                }
            }
        }
    };
}

compare_tests!(MlKem512, mlkem_native_rs::mlkem512, mlkem512);
compare_tests!(MlKem768, mlkem_native_rs::mlkem768, mlkem768);
compare_tests!(MlKem1024, mlkem_native_rs::mlkem1024, mlkem1024);
