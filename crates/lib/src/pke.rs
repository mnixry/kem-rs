//! IND-CPA public-key encryption -- the inner PKE scheme used by ML-KEM.

use sha3::digest::XofReader;

use crate::{
    hash,
    math::{poly::Poly, polyvec::PolyVec, sample},
    params::{MlKemParams, N, SYMBYTES},
};

fn getnoise_eta(eta: usize, seed: &[u8; SYMBYTES], nonce: u8) -> Poly {
    let mut p = Poly::zero();
    match eta {
        2 => {
            let mut buf = [0u8; 2 * N / 4];
            hash::prf(seed, nonce, &mut buf);
            sample::cbd2(&mut p.coeffs, &buf);
        }
        3 => {
            let mut buf = [0u8; 3 * N / 4];
            hash::prf(seed, nonce, &mut buf);
            sample::cbd3(&mut p.coeffs, &buf);
        }
        _ => unreachable!(),
    }
    p
}

/// Sample the KxK public matrix `A` from seed `rho` using SHAKE-128.
/// If `transposed`, produces `A^T` for encryption.
fn gen_matrix<const K: usize>(a: &mut [PolyVec<K>], seed: &[u8; SYMBYTES], transposed: bool) {
    for (i, a_row) in a.iter_mut().enumerate() {
        for (j, poly) in a_row.polys.iter_mut().enumerate() {
            let (x, y) = if transposed {
                (i as u8, j as u8)
            } else {
                (j as u8, i as u8)
            };
            let mut xof = hash::xof_absorb(seed, x, y);
            sample::rej_uniform(&mut poly.coeffs, |buf| xof.read(buf));
        }
    }
}

// -- IND-CPA key generation --------------------------------------------------

pub(crate) fn indcpa_keypair_derand<P: MlKemParams>(
    pk_bytes: &mut [u8], sk_bytes: &mut [u8], coins: &[u8; SYMBYTES],
) {
    match P::K {
        2 => indcpa_keypair_inner::<P, 2>(pk_bytes, sk_bytes, coins),
        3 => indcpa_keypair_inner::<P, 3>(pk_bytes, sk_bytes, coins),
        4 => indcpa_keypair_inner::<P, 4>(pk_bytes, sk_bytes, coins),
        _ => unreachable!(),
    }
}

fn indcpa_keypair_inner<P: MlKemParams, const K: usize>(
    pk_bytes: &mut [u8], sk_bytes: &mut [u8], coins: &[u8; SYMBYTES],
) {
    // G(d || k) -> (rho || sigma)
    let mut g_input = [0u8; SYMBYTES + 1];
    g_input[..SYMBYTES].copy_from_slice(coins);
    g_input[SYMBYTES] = K as u8;
    let buf = hash::hash_g(g_input);
    let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
    let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

    let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
    gen_matrix::<K>(&mut a, &public_seed, false);

    let mut nonce: u8 = 0;
    let mut skpv = PolyVec::<K>::zero();
    for p in &mut skpv.polys {
        *p = getnoise_eta(P::ETA1, &noise_seed, nonce);
        nonce += 1;
    }
    let mut e = PolyVec::<K>::zero();
    for p in &mut e.polys {
        *p = getnoise_eta(P::ETA1, &noise_seed, nonce);
        nonce += 1;
    }

    skpv.ntt();
    skpv.reduce();
    e.ntt();

    // t = A * s + e (NTT domain)
    let mut pkpv = PolyVec::<K>::zero();
    for (pk_poly, a_row) in pkpv.polys.iter_mut().zip(a.iter()) {
        PolyVec::basemul_acc_montgomery(pk_poly, a_row, &skpv);
        pk_poly.tomont();
    }
    pkpv += &e;
    pkpv.reduce();

    // pk = Encode_12(t) || rho, sk = Encode_12(s)
    pkpv.tobytes(&mut pk_bytes[..P::POLYVEC_BYTES]);
    pk_bytes[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES].copy_from_slice(&public_seed);
    skpv.tobytes(&mut sk_bytes[..P::INDCPA_SK_BYTES]);
}

// -- IND-CPA encryption ------------------------------------------------------

pub(crate) fn indcpa_enc<P: MlKemParams>(
    ct_bytes: &mut [u8], m: &[u8; SYMBYTES], pk_bytes: &[u8], coins: &[u8; SYMBYTES],
) {
    match P::K {
        2 => indcpa_enc_inner::<P, 2>(ct_bytes, m, pk_bytes, coins),
        3 => indcpa_enc_inner::<P, 3>(ct_bytes, m, pk_bytes, coins),
        4 => indcpa_enc_inner::<P, 4>(ct_bytes, m, pk_bytes, coins),
        _ => unreachable!(),
    }
}

fn indcpa_enc_inner<P: MlKemParams, const K: usize>(
    ct_bytes: &mut [u8], m: &[u8; SYMBYTES], pk_bytes: &[u8], coins: &[u8; SYMBYTES],
) {
    let pkpv = PolyVec::<K>::frombytes(&pk_bytes[..P::POLYVEC_BYTES]);
    let seed: [u8; SYMBYTES] = pk_bytes[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES]
        .try_into()
        .unwrap();
    let k = Poly::frommsg(m);

    let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
    gen_matrix::<K>(&mut at, &seed, true);

    let mut nonce: u8 = 0;
    let mut sp = PolyVec::<K>::zero();
    for p in &mut sp.polys {
        *p = getnoise_eta(P::ETA1, coins, nonce);
        nonce += 1;
    }
    let mut ep = PolyVec::<K>::zero();
    for p in &mut ep.polys {
        *p = getnoise_eta(P::ETA2, coins, nonce);
        nonce += 1;
    }
    let epp = getnoise_eta(P::ETA2, coins, nonce);

    sp.ntt();

    // u = A^T * r + e1
    let mut b = PolyVec::<K>::zero();
    for (b_poly, at_row) in b.polys.iter_mut().zip(at.iter()) {
        PolyVec::basemul_acc_montgomery(b_poly, at_row, &sp);
    }

    // v = t^T * r + e2 + Decompress_1(m)
    let mut v = Poly::zero();
    PolyVec::basemul_acc_montgomery(&mut v, &pkpv, &sp);

    b.invntt_tomont();
    v.invntt_tomont();

    b += &ep;
    v += &epp;
    v += &k;

    b.reduce();
    v.reduce();

    b.compress(&mut ct_bytes[..P::POLYVEC_COMPRESSED_BYTES], P::D_U);
    v.compress(
        &mut ct_bytes[P::POLYVEC_COMPRESSED_BYTES..P::INDCPA_BYTES],
        P::D_V,
    );
}

// -- IND-CPA decryption ------------------------------------------------------

pub(crate) fn indcpa_dec<P: MlKemParams>(m: &mut [u8; SYMBYTES], ct_bytes: &[u8], sk_bytes: &[u8]) {
    match P::K {
        2 => indcpa_dec_inner::<P, 2>(m, ct_bytes, sk_bytes),
        3 => indcpa_dec_inner::<P, 3>(m, ct_bytes, sk_bytes),
        4 => indcpa_dec_inner::<P, 4>(m, ct_bytes, sk_bytes),
        _ => unreachable!(),
    }
}

fn indcpa_dec_inner<P: MlKemParams, const K: usize>(
    m: &mut [u8; SYMBYTES], ct_bytes: &[u8], sk_bytes: &[u8],
) {
    let b = PolyVec::<K>::decompress(&ct_bytes[..P::POLYVEC_COMPRESSED_BYTES], P::D_U);
    let v = Poly::decompress(
        &ct_bytes[P::POLYVEC_COMPRESSED_BYTES..P::INDCPA_BYTES],
        P::D_V,
    );
    let skpv = PolyVec::<K>::frombytes(&sk_bytes[..P::INDCPA_SK_BYTES]);

    let mut b_ntt = b;
    b_ntt.ntt();

    let mut mp = Poly::zero();
    PolyVec::basemul_acc_montgomery(&mut mp, &skpv, &b_ntt);
    mp.invntt_tomont();

    // m' = v - s^T * u
    let mut msg = &v - &mp;
    msg.reduce();
    *m = msg.tomsg();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{MlKem512, MlKem768, MlKem1024};

    /// Manual pipeline: keygen + encrypt zero message + decrypt, checking
    /// noise.
    fn manual_pipeline<const K: usize>(eta1: usize, eta2: usize) {
        let coins = [42u8; SYMBYTES];

        let mut g_input = [0u8; SYMBYTES + 1];
        g_input[..SYMBYTES].copy_from_slice(&coins);
        g_input[SYMBYTES] = K as u8;
        let buf = hash::hash_g(g_input);
        let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
        let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

        let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut a, &public_seed, false);

        let mut nonce = 0u8;
        let mut s = PolyVec::<K>::zero();
        for p in &mut s.polys {
            *p = getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        let mut e = PolyVec::<K>::zero();
        for p in &mut e.polys {
            *p = getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }

        s.ntt();
        s.reduce();
        e.ntt();

        let mut t = PolyVec::<K>::zero();
        for (t_poly, a_row) in t.polys.iter_mut().zip(a.iter()) {
            PolyVec::basemul_acc_montgomery(t_poly, a_row, &s);
            t_poly.tomont();
        }
        t += &e;
        t.reduce();

        // Encrypt zero message, then decrypt
        let enc_coins = [7u8; SYMBYTES];
        let zero_msg = [0u8; SYMBYTES];
        let k_poly = Poly::frommsg(&zero_msg);

        let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut at, &public_seed, true);

        let mut enc_nonce = 0u8;
        let mut r = PolyVec::<K>::zero();
        for p in &mut r.polys {
            *p = getnoise_eta(eta1, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let mut e1 = PolyVec::<K>::zero();
        for p in &mut e1.polys {
            *p = getnoise_eta(eta2, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let e2 = getnoise_eta(eta2, &enc_coins, enc_nonce);

        r.ntt();

        let mut u = PolyVec::<K>::zero();
        for (u_poly, at_row) in u.polys.iter_mut().zip(at.iter()) {
            PolyVec::basemul_acc_montgomery(u_poly, at_row, &r);
        }
        let mut v = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut v, &t, &r);

        u.invntt_tomont();
        v.invntt_tomont();
        u += &e1;
        v += &e2;
        v += &k_poly;
        u.reduce();
        v.reduce();

        // Decrypt (no compression)
        let mut u_ntt = u.clone();
        u_ntt.ntt();
        let mut mp = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut mp, &s, &u_ntt);
        mp.invntt_tomont();

        let mut result = &v - &mp;
        result.reduce();

        let q = crate::params::Q as i32;
        let max_coeff = result
            .coeffs
            .iter()
            .map(|&c| {
                let cv = c as i32;
                if cv > q / 2 {
                    cv - q
                } else if cv < -q / 2 {
                    cv + q
                } else {
                    cv
                }
            })
            .map(|c| c.abs())
            .max()
            .unwrap();
        assert!(
            max_coeff < q / 4,
            "noise too large: {max_coeff} >= {}",
            q / 4
        );

        let recovered = result.tomsg();
        assert_eq!(recovered, zero_msg, "zero-message recovery failed");
    }

    #[test]
    fn manual_pipeline_k2() {
        manual_pipeline::<2>(3, 2);
    }
    #[test]
    fn manual_pipeline_k3() {
        manual_pipeline::<3>(2, 2);
    }
    #[test]
    fn manual_pipeline_k4() {
        manual_pipeline::<4>(2, 2);
    }

    /// Same as manual_pipeline but with serialization/deserialization
    /// roundtrip.
    fn serialized_manual_pipeline<const K: usize>(eta1: usize, eta2: usize) {
        let coins = [42u8; SYMBYTES];
        let polyvec_bytes = K * crate::params::POLYBYTES;

        let mut g_input = [0u8; SYMBYTES + 1];
        g_input[..SYMBYTES].copy_from_slice(&coins);
        g_input[SYMBYTES] = K as u8;
        let buf = hash::hash_g(g_input);
        let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
        let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

        let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut a, &public_seed, false);

        let mut nonce = 0u8;
        let mut s = PolyVec::<K>::zero();
        for p in &mut s.polys {
            *p = getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        let mut e = PolyVec::<K>::zero();
        for p in &mut e.polys {
            *p = getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        s.ntt();
        s.reduce();
        e.ntt();

        let mut t = PolyVec::<K>::zero();
        for (t_poly, a_row) in t.polys.iter_mut().zip(a.iter()) {
            PolyVec::basemul_acc_montgomery(t_poly, a_row, &s);
            t_poly.tomont();
        }
        t += &e;
        t.reduce();

        // Serialize / deserialize
        let mut pk_buf = vec![0u8; polyvec_bytes + SYMBYTES];
        t.tobytes(&mut pk_buf[..polyvec_bytes]);
        pk_buf[polyvec_bytes..].copy_from_slice(&public_seed);
        let mut sk_buf = vec![0u8; polyvec_bytes];
        s.tobytes(&mut sk_buf);

        let t2 = PolyVec::<K>::frombytes(&pk_buf[..polyvec_bytes]);
        let s2 = PolyVec::<K>::frombytes(&sk_buf);

        // Verify pk roundtrip
        for k in 0..K {
            for j in 0..crate::params::N {
                let orig = crate::math::pack::csubq(t.polys[k].coeffs[j]);
                let deser = t2.polys[k].coeffs[j] as u16;
                assert_eq!(orig, deser, "t serialize mismatch poly {k} coeff {j}");
            }
        }

        // Encrypt using deserialized pk
        let enc_coins = [7u8; SYMBYTES];
        let zero_msg = [0u8; SYMBYTES];
        let k_poly = Poly::frommsg(&zero_msg);

        let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut at, &public_seed, true);

        let mut enc_nonce = 0u8;
        let mut r = PolyVec::<K>::zero();
        for p in &mut r.polys {
            *p = getnoise_eta(eta1, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let mut e1 = PolyVec::<K>::zero();
        for p in &mut e1.polys {
            *p = getnoise_eta(eta2, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let e2_poly = getnoise_eta(eta2, &enc_coins, enc_nonce);

        r.ntt();

        // Verify pk basemul consistency
        {
            let mut v_direct = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut v_direct, &t, &r);
            let mut v_deser = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut v_deser, &t2, &r);
            let q = crate::params::Q as i32;
            for j in 0..crate::params::N {
                let (d, ds) = (v_direct.coeffs[j] as i32, v_deser.coeffs[j] as i32);
                let diff = ((d - ds) % q + q) % q;
                let diff = diff.min(q - diff);
                assert_eq!(diff, 0, "pk basemul diff at coeff {j}");
            }
        }

        let mut u = PolyVec::<K>::zero();
        for (u_poly, at_row) in u.polys.iter_mut().zip(at.iter()) {
            PolyVec::basemul_acc_montgomery(u_poly, at_row, &r);
        }
        let mut v = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut v, &t2, &r);

        u.invntt_tomont();
        v.invntt_tomont();
        u += &e1;
        v += &e2_poly;
        v += &k_poly;
        u.reduce();
        v.reduce();

        // Decrypt using deserialized sk
        let mut u_ntt = u.clone();
        u_ntt.ntt();

        // Verify sk basemul consistency
        {
            let mut mp_direct = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut mp_direct, &s, &u_ntt);
            let mut mp_deser = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut mp_deser, &s2, &u_ntt);
            let q = crate::params::Q as i32;
            for j in 0..crate::params::N {
                let (d, ds) = (mp_direct.coeffs[j] as i32, mp_deser.coeffs[j] as i32);
                let diff = ((d - ds) % q + q) % q;
                let diff = diff.min(q - diff);
                assert_eq!(diff, 0, "sk basemul diff at coeff {j}");
            }
        }

        let mut mp = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut mp, &s2, &u_ntt);
        mp.invntt_tomont();

        let mut result = &v - &mp;
        result.reduce();
        let recovered = result.tomsg();
        assert_eq!(
            recovered, zero_msg,
            "serialized pipeline: zero-message recovery failed"
        );
    }

    #[test]
    fn serialized_manual_k2() {
        serialized_manual_pipeline::<2>(3, 2);
    }
    #[test]
    fn serialized_manual_k3() {
        serialized_manual_pipeline::<3>(2, 2);
    }

    fn indcpa_roundtrip<P: MlKemParams>() {
        let seed = [42u8; SYMBYTES];
        let mut pk = vec![0u8; P::INDCPA_PK_BYTES];
        let mut sk = vec![0u8; P::INDCPA_SK_BYTES];
        indcpa_keypair_derand::<P>(&mut pk, &mut sk, &seed);

        let msg = [0xAB; SYMBYTES];
        let coins = [7u8; SYMBYTES];
        let mut ct = vec![0u8; P::INDCPA_BYTES];
        indcpa_enc::<P>(&mut ct, &msg, &pk, &coins);

        let mut recovered = [0u8; SYMBYTES];
        indcpa_dec::<P>(&mut recovered, &ct, &sk);
        assert_eq!(msg, recovered, "IND-CPA roundtrip failed");
    }

    #[test]
    fn indcpa_roundtrip_512() {
        indcpa_roundtrip::<MlKem512>();
    }
    #[test]
    fn indcpa_roundtrip_768() {
        indcpa_roundtrip::<MlKem768>();
    }
    #[test]
    fn indcpa_roundtrip_1024() {
        indcpa_roundtrip::<MlKem1024>();
    }
}
