//! IND-CPA public-key encryption (K-PKE) for ML-KEM.
//!
//! Generic over [`ParameterSet`] with no runtime dispatch. Noise via sealed
//! [`CbdWidth`]; compression via sealed [`CompressWidth`].

use kem_math::{ByteArray, Polynomial, SYMBYTES};
use zerocopy::transmute_ref;

use crate::{
    params::ParameterSet,
    types::{Ciphertext, PublicKey, Sym2},
};

/// Deterministic IND-CPA key generation.
///
/// Returns `(pk, indcpa_sk)` where `indcpa_sk` is the serialised NTT secret
/// vector.
pub(crate) fn indcpa_keypair_derand<P: ParameterSet>(
    coins: &[u8; SYMBYTES],
) -> (PublicKey<P>, P::PolyVecArray) {
    let mut g_input = [0u8; SYMBYTES + 1];
    g_input[..SYMBYTES].copy_from_slice(coins);
    g_input[SYMBYTES] = P::K as u8;
    let buf = kem_hash::hash_g(g_input);
    let Sym2(public_seed, noise_seed) = transmute_ref!(&buf);

    let a_hat = P::gen_matrix(public_seed, false);

    let mut nonce = 0u8;
    let mut s_hat = P::sample_noise_eta1(noise_seed, &mut nonce);
    P::reduce_ntt_vec(&mut s_hat);
    let e_hat = P::sample_noise_eta1(noise_seed, &mut nonce);

    let mut t_hat = P::mat_mul_vec_tomont(&a_hat, &s_hat);
    t_hat = P::add_ntt_vecs(&t_hat, &e_hat);
    P::reduce_ntt_vec(&mut t_hat);

    let mut polyvec = P::PolyVecArray::zeroed();
    P::ntt_vec_to_bytes(&t_hat, polyvec.as_mut());

    let pk = PublicKey {
        polyvec,
        rho: *public_seed,
    };

    let mut sk = P::PolyVecArray::zeroed();
    P::ntt_vec_to_bytes(&s_hat, sk.as_mut());

    (pk, sk)
}

/// IND-CPA encryption.
pub(crate) fn indcpa_enc<P: ParameterSet>(
    pk: &PublicKey<P>, m: &[u8; SYMBYTES], coins: &[u8; SYMBYTES],
) -> Ciphertext<P> {
    let t_hat = P::ntt_vec_from_bytes(pk.polyvec.as_ref());

    let a_hat_t = P::gen_matrix(&pk.rho, true);

    let mut nonce = 0u8;
    let r_hat = P::sample_noise_eta1(coins, &mut nonce);
    let e1 = P::sample_noise_eta2(coins, &mut nonce);

    let e2 = Polynomial::sample_cbd::<P::Eta2>(&kem_hash::prf::<P::Eta2>(coins, nonce));

    let u_hat = P::mat_mul_vec(&a_hat_t, &r_hat);
    let u_std = P::inv_ntt_vec(u_hat);
    let mut u = P::add_vecs(&u_std, &e1);
    P::reduce_vec(&mut u);

    let v_ntt = P::inner_product(&t_hat, &r_hat);
    let v_std = v_ntt.ntt_inverse();
    let msg_poly = Polynomial::from_message(m);
    let mut v = &(&v_std + &e2) + &msg_poly;
    v.reduce();

    let mut ct = Ciphertext {
        u_compressed: P::PolyVecCompArray::zeroed(),
        v_compressed: P::PolyCompArray::zeroed(),
    };
    P::vec_compress(&u, AsMut::<[u8]>::as_mut(&mut ct.u_compressed));
    v.compress::<P::Dv>(AsMut::<[u8]>::as_mut(&mut ct.v_compressed));

    ct
}

/// IND-CPA decryption.
pub(crate) fn indcpa_dec<P: ParameterSet>(
    ct: &Ciphertext<P>, indcpa_sk: &P::PolyVecArray,
) -> [u8; SYMBYTES] {
    let u = P::vec_decompress(ct.u_compressed.as_ref());
    let v = Polynomial::decompress::<P::Dv>(ct.v_compressed.as_ref());

    let s_hat = P::ntt_vec_from_bytes(indcpa_sk.as_ref());
    let u_hat = P::ntt_vec(u);
    let su = P::inner_product(&s_hat, &u_hat).ntt_inverse();
    let mut mp = &v - &su;
    mp.reduce();
    mp.to_message()
}
