//! IND-CPA public-key encryption (K-PKE) for ML-KEM.
//!
//! Generic over [`ParameterSet`] with no runtime dispatch. Noise via sealed
//! [`CbdWidth`]; compression via sealed [`CompressWidth`].

use kem_math::{ByteArray, CbdWidthParams, Polynomial, SYMBYTES};

use crate::params::ParameterSet;

/// Deterministic IND-CPA key generation.
pub(crate) fn indcpa_keypair_derand<P: ParameterSet>(
    coins: &[u8; SYMBYTES],
) -> (P::PkArray, P::PkArray) {
    let mut g_input = [0u8; SYMBYTES + 1];
    g_input[..SYMBYTES].copy_from_slice(coins);
    g_input[SYMBYTES] = P::K as u8;
    let buf = kem_hash::hash_g(g_input);
    let public_seed: &[u8; SYMBYTES] = buf[..SYMBYTES].first_chunk().expect("buf is 64 bytes");
    let noise_seed: &[u8; SYMBYTES] = buf[SYMBYTES..].first_chunk().expect("buf is 64 bytes");

    let a_hat = P::gen_matrix(public_seed, false);

    let mut nonce = 0u8;
    let mut s_hat = P::sample_noise_eta1(noise_seed, &mut nonce);
    P::reduce_ntt_vec(&mut s_hat);
    let e_hat = P::sample_noise_eta1(noise_seed, &mut nonce);

    let mut t_hat = P::mat_mul_vec_tomont(&a_hat, &s_hat);
    t_hat = P::add_ntt_vecs(&t_hat, &e_hat);
    P::reduce_ntt_vec(&mut t_hat);

    let mut pk = P::PkArray::zeroed();
    P::ntt_vec_to_bytes(&t_hat, &mut pk.as_mut()[..P::POLYVEC_BYTES]);
    pk.as_mut()[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES].copy_from_slice(public_seed);

    let mut sk = P::PkArray::zeroed();
    P::ntt_vec_to_bytes(&s_hat, &mut sk.as_mut()[..P::INDCPA_SK_BYTES]);

    (pk, sk)
}

/// IND-CPA encryption.
pub(crate) fn indcpa_enc<P: ParameterSet>(
    pk_bytes: &[u8], m: &[u8; SYMBYTES], coins: &[u8; SYMBYTES],
) -> P::CtArray {
    let t_hat = P::ntt_vec_from_bytes(&pk_bytes[..P::POLYVEC_BYTES]);
    let rho: &[u8; SYMBYTES] = pk_bytes[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES]
        .first_chunk()
        .expect("pk has rho");

    let a_hat_t = P::gen_matrix(rho, true);

    let mut nonce = 0u8;
    let r_hat = P::sample_noise_eta1(coins, &mut nonce);
    let e1 = P::sample_noise_eta2(coins, &mut nonce);

    let buf_len = <<P as ParameterSet>::Eta2 as CbdWidthParams>::BUF_BYTES;
    let mut e2_buf = [0u8; 192];
    kem_hash::prf(coins, nonce, &mut e2_buf[..buf_len]);
    let e2 = Polynomial::sample_cbd::<P::Eta2>(&e2_buf[..buf_len]);

    let u_hat = P::mat_mul_vec(&a_hat_t, &r_hat);
    let u_std = P::inv_ntt_vec(u_hat);
    let mut u = P::add_vecs(&u_std, &e1);
    P::reduce_vec(&mut u);

    let v_ntt = P::inner_product(&t_hat, &r_hat);
    let v_std = v_ntt.ntt_inverse();
    let msg_poly = Polynomial::from_message(m);
    let mut v = &(&v_std + &e2) + &msg_poly;
    v.reduce();

    let mut ct = P::CtArray::zeroed();
    P::vec_compress(&u, &mut ct.as_mut()[..P::POLYVEC_COMPRESSED_BYTES]);
    v.compress::<P::Dv>(
        &mut ct.as_mut()
            [P::POLYVEC_COMPRESSED_BYTES..P::POLYVEC_COMPRESSED_BYTES + P::POLY_COMPRESSED_BYTES],
    );

    ct
}

/// IND-CPA decryption.
pub(crate) fn indcpa_dec<P: ParameterSet>(ct_bytes: &[u8], sk_bytes: &[u8]) -> [u8; SYMBYTES] {
    let u = P::vec_decompress(&ct_bytes[..P::POLYVEC_COMPRESSED_BYTES]);
    let v = Polynomial::decompress::<P::Dv>(
        &ct_bytes
            [P::POLYVEC_COMPRESSED_BYTES..P::POLYVEC_COMPRESSED_BYTES + P::POLY_COMPRESSED_BYTES],
    );

    let s_hat = P::ntt_vec_from_bytes(&sk_bytes[..P::INDCPA_SK_BYTES]);
    let u_hat = P::ntt_vec(u);
    let su = P::inner_product(&s_hat, &u_hat).ntt_inverse();
    let mut mp = &v - &su;
    mp.reduce();
    mp.to_message()
}
