//! Deterministic and randomized ML-KEM behavior checks.
#![feature(macro_metavar_expr_concat)]

use kem_rs::{
    Ciphertext, MlKem512, MlKem768, MlKem1024, MlKemParams, decapsulate, encapsulate,
    encapsulate_derand, keypair, keypair_derand, params::ByteArray,
};
use rand_core::UnwrapErr;

fn fixed_keygen_coins(variant: u8) -> [u8; 64] {
    core::array::from_fn(|i| (i as u8).wrapping_add(variant.wrapping_mul(37)))
}

fn fixed_enc_coins(variant: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(variant.wrapping_mul(53)))
}

fn check_kem_roundtrip<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(0);
    let enc_coins = fixed_enc_coins(0);
    let (pk, sk) = keypair_derand::<P>(&kp_coins);
    let (ct, ss_enc) = encapsulate_derand::<P>(&pk, &enc_coins);
    let ss_dec = decapsulate::<P>(&ct, &sk);
    assert_eq!(ss_enc.as_ref(), ss_dec.as_ref());
}

fn check_determinism<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(1);
    let enc_coins = fixed_enc_coins(1);

    let (pk1, sk1) = keypair_derand::<P>(&kp_coins);
    let (pk2, sk2) = keypair_derand::<P>(&kp_coins);
    assert_eq!(pk1.as_ref(), pk2.as_ref());
    assert_eq!(sk1.as_ref(), sk2.as_ref());

    let (ct1, ss1) = encapsulate_derand::<P>(&pk1, &enc_coins);
    let (ct2, ss2) = encapsulate_derand::<P>(&pk2, &enc_coins);
    assert_eq!(ct1.as_ref(), ct2.as_ref());
    assert_eq!(ss1.as_ref(), ss2.as_ref());

    let ss_dec1 = decapsulate::<P>(&ct1, &sk1);
    let ss_dec2 = decapsulate::<P>(&ct2, &sk2);
    assert_eq!(ss_dec1.as_ref(), ss_dec2.as_ref());
    assert_eq!(ss1.as_ref(), ss_dec1.as_ref());
}

fn check_implicit_rejection<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(2);
    let enc_coins = fixed_enc_coins(2);

    let (pk, sk) = keypair_derand::<P>(&kp_coins);
    let (ct, ss_good) = encapsulate_derand::<P>(&pk, &enc_coins);

    let mut bad_ct_bytes = P::CtArray::zeroed();
    bad_ct_bytes.as_mut().copy_from_slice(ct.as_ref());
    bad_ct_bytes.as_mut()[0] ^= 0xFF;
    let bad_ct = Ciphertext::<P>::from(&bad_ct_bytes);

    let ss_bad = decapsulate::<P>(&bad_ct, &sk);
    assert_ne!(ss_good.as_ref(), ss_bad.as_ref());

    let ss_bad2 = decapsulate::<P>(&bad_ct, &sk);
    assert_eq!(ss_bad.as_ref(), ss_bad2.as_ref());
}

fn check_wrong_secret_key<P: MlKemParams>() {
    let (pk, _) = keypair_derand::<P>(&fixed_keygen_coins(3));
    let (_, wrong_sk) = keypair_derand::<P>(&fixed_keygen_coins(4));
    let (ct, ss_enc) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(3));
    let ss_wrong = decapsulate::<P>(&ct, &wrong_sk);
    assert_ne!(ss_enc.as_ref(), ss_wrong.as_ref());
}

fn check_sizes<P: MlKemParams>() {
    let (pk, sk) = keypair_derand::<P>(&fixed_keygen_coins(5));
    let (ct, _) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(5));
    assert_eq!(pk.as_ref().len(), P::PK_BYTES);
    assert_eq!(sk.as_ref().len(), P::SK_BYTES);
    assert_eq!(ct.as_ref().len(), P::CT_BYTES);
}

fn check_randomized_roundtrip<P: MlKemParams>() {
    let mut rng = UnwrapErr(getrandom::SysRng);
    let (pk, sk) = keypair::<P>(&mut rng);
    let (ct, ss_enc) = encapsulate::<P>(&pk, &mut rng);
    let ss_dec = decapsulate::<P>(&ct, &sk);
    assert_eq!(ss_enc.as_ref(), ss_dec.as_ref());
}

fn check_different_encaps<P: MlKemParams>() {
    let (pk, sk) = keypair_derand::<P>(&fixed_keygen_coins(6));
    let (ct1, ss1) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(10));
    let (ct2, ss2) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(11));

    assert_ne!(ct1.as_ref(), ct2.as_ref());
    assert_ne!(ss1.as_ref(), ss2.as_ref());
    assert_eq!(ss1.as_ref(), decapsulate::<P>(&ct1, &sk).as_ref());
    assert_eq!(ss2.as_ref(), decapsulate::<P>(&ct2, &sk).as_ref());
}

macro_rules! check_for_param_set {
  ($fn_name:ident, $($ty:ident),*) => {
    $(
      #[test]
      #[allow(non_snake_case)]
      fn ${ concat($fn_name, _, $ty) }() {
        $fn_name::<$ty>();
      }
    )*
  }
}

check_for_param_set!(check_kem_roundtrip, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_determinism, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_implicit_rejection, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_wrong_secret_key, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_sizes, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_randomized_roundtrip, MlKem512, MlKem768, MlKem1024);
check_for_param_set!(check_different_encaps, MlKem512, MlKem768, MlKem1024);
