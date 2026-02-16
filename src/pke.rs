//! IND-CPA public-key encryption — the inner PKE scheme used by ML-KEM.
//!
//! Not exposed directly; called by [`crate::kem`].

use crate::hash;
use crate::math::{poly::Poly, polyvec::PolyVec, sample};
use crate::params::{MlKemParams, SYMBYTES};

// ---------------------------------------------------------------------------
// Matrix generation
// ---------------------------------------------------------------------------

/// Sample the K×K public matrix A from seed ρ using SHAKE-128.
///
/// If `transposed`, indices are swapped (produces Aᵀ for encryption).
fn gen_matrix<const K: usize>(a: &mut [PolyVec<K>], seed: &[u8; SYMBYTES], transposed: bool) {
    for i in 0..K {
        for j in 0..K {
            let (x, y) = if transposed {
                (i as u8, j as u8)
            } else {
                (j as u8, i as u8)
            };
            let mut xof = hash::xof_absorb(seed, x, y);
            sample::rej_uniform(&mut a[i].polys[j].coeffs, &mut xof);
        }
    }
}

// ---------------------------------------------------------------------------
// IND-CPA key generation (deterministic)
// ---------------------------------------------------------------------------

/// Deterministic IND-CPA keypair generation.
///
/// `coins` is 32 bytes of randomness (the seed `d` in FIPS 203).
/// Writes the public key to `pk_bytes` and the IND-CPA secret key to `sk_bytes`.
pub(crate) fn indcpa_keypair_derand<P: MlKemParams>(
    pk_bytes: &mut [u8],
    sk_bytes: &mut [u8],
    coins: &[u8; SYMBYTES],
) {
    match P::K {
        2 => indcpa_keypair_inner::<P, 2>(pk_bytes, sk_bytes, coins),
        3 => indcpa_keypair_inner::<P, 3>(pk_bytes, sk_bytes, coins),
        4 => indcpa_keypair_inner::<P, 4>(pk_bytes, sk_bytes, coins),
        _ => unreachable!(),
    }
}

fn indcpa_keypair_inner<P: MlKemParams, const K: usize>(
    pk_bytes: &mut [u8],
    sk_bytes: &mut [u8],
    coins: &[u8; SYMBYTES],
) {
    // G(d ‖ k) → (ρ ‖ σ)  (FIPS 203: k is appended before hashing)
    let mut g_input = [0u8; SYMBYTES + 1];
    g_input[..SYMBYTES].copy_from_slice(coins);
    g_input[SYMBYTES] = K as u8;
    let buf = hash::hash_g(&g_input);
    let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
    let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

    // Sample matrix A
    let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
    gen_matrix::<K>(&mut a, &public_seed, false);

    // Sample secret vector s and error vector e
    let mut nonce: u8 = 0;
    let mut skpv = PolyVec::<K>::zero();
    for i in 0..K {
        skpv.polys[i] = Poly::getnoise_eta(P::ETA1, &noise_seed, nonce);
        nonce += 1;
    }
    let mut e = PolyVec::<K>::zero();
    for i in 0..K {
        e.polys[i] = Poly::getnoise_eta(P::ETA1, &noise_seed, nonce);
        nonce += 1;
    }

    // NTT(s), NTT(e)
    skpv.ntt();
    skpv.reduce();
    e.ntt();

    // t = A · s + e  (in NTT domain)
    let mut pkpv = PolyVec::<K>::zero();
    for i in 0..K {
        PolyVec::basemul_acc_montgomery(&mut pkpv.polys[i], &a[i], &skpv);
        pkpv.polys[i].tomont();
    }
    pkpv.add_assign(&e);
    pkpv.reduce();

    // Pack: pk = (Encode₁₂(t) ‖ ρ),  sk = Encode₁₂(s)
    pkpv.tobytes(&mut pk_bytes[..P::POLYVEC_BYTES]);
    pk_bytes[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES].copy_from_slice(&public_seed);
    skpv.tobytes(&mut sk_bytes[..P::INDCPA_SK_BYTES]);
}

// ---------------------------------------------------------------------------
// IND-CPA encryption (deterministic)
// ---------------------------------------------------------------------------

/// Deterministic IND-CPA encryption.
///
/// Encrypts message `m` under public key `pk_bytes` using coins `coins`.
pub(crate) fn indcpa_enc<P: MlKemParams>(
    ct_bytes: &mut [u8],
    m: &[u8; SYMBYTES],
    pk_bytes: &[u8],
    coins: &[u8; SYMBYTES],
) {
    match P::K {
        2 => indcpa_enc_inner::<P, 2>(ct_bytes, m, pk_bytes, coins),
        3 => indcpa_enc_inner::<P, 3>(ct_bytes, m, pk_bytes, coins),
        4 => indcpa_enc_inner::<P, 4>(ct_bytes, m, pk_bytes, coins),
        _ => unreachable!(),
    }
}

fn indcpa_enc_inner<P: MlKemParams, const K: usize>(
    ct_bytes: &mut [u8],
    m: &[u8; SYMBYTES],
    pk_bytes: &[u8],
    coins: &[u8; SYMBYTES],
) {
    // Unpack pk
    let pkpv = PolyVec::<K>::frombytes(&pk_bytes[..P::POLYVEC_BYTES]);
    let seed: [u8; SYMBYTES] = pk_bytes[P::POLYVEC_BYTES..P::INDCPA_PK_BYTES]
        .try_into()
        .unwrap();

    let k = Poly::frommsg(m);

    // Sample Aᵀ (transposed for encryption)
    let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
    gen_matrix::<K>(&mut at, &seed, true);

    // Sample r, e₁, e₂
    let mut nonce: u8 = 0;
    let mut sp = PolyVec::<K>::zero();
    for i in 0..K {
        sp.polys[i] = Poly::getnoise_eta(P::ETA1, coins, nonce);
        nonce += 1;
    }
    let mut ep = PolyVec::<K>::zero();
    for i in 0..K {
        ep.polys[i] = Poly::getnoise_eta(P::ETA2, coins, nonce);
        nonce += 1;
    }
    let epp = Poly::getnoise_eta(P::ETA2, coins, nonce);

    // NTT(r)
    sp.ntt();

    // u = Aᵀ · r + e₁
    let mut b = PolyVec::<K>::zero();
    for i in 0..K {
        PolyVec::basemul_acc_montgomery(&mut b.polys[i], &at[i], &sp);
    }

    // v = tᵀ · r + e₂ + Decompress₁(m)
    let mut v = Poly::zero();
    PolyVec::basemul_acc_montgomery(&mut v, &pkpv, &sp);

    b.invntt_tomont();
    v.invntt_tomont();

    b.add_assign(&ep);
    v.add_assign(&epp);
    v.add_assign(&k);

    b.reduce();
    v.reduce();

    // Pack ciphertext: c = (Compress_{d_u}(u) ‖ Compress_{d_v}(v))
    b.compress(&mut ct_bytes[..P::POLYVEC_COMPRESSED_BYTES], P::D_U);
    v.compress(
        &mut ct_bytes[P::POLYVEC_COMPRESSED_BYTES..P::INDCPA_BYTES],
        P::D_V,
    );
}

// ---------------------------------------------------------------------------
// IND-CPA decryption
// ---------------------------------------------------------------------------

/// IND-CPA decryption: recovers the message from ciphertext and secret key.
pub(crate) fn indcpa_dec<P: MlKemParams>(
    m: &mut [u8; SYMBYTES],
    ct_bytes: &[u8],
    sk_bytes: &[u8],
) {
    match P::K {
        2 => indcpa_dec_inner::<P, 2>(m, ct_bytes, sk_bytes),
        3 => indcpa_dec_inner::<P, 3>(m, ct_bytes, sk_bytes),
        4 => indcpa_dec_inner::<P, 4>(m, ct_bytes, sk_bytes),
        _ => unreachable!(),
    }
}

fn indcpa_dec_inner<P: MlKemParams, const K: usize>(
    m: &mut [u8; SYMBYTES],
    ct_bytes: &[u8],
    sk_bytes: &[u8],
) {
    // Unpack ciphertext
    let b = PolyVec::<K>::decompress(&ct_bytes[..P::POLYVEC_COMPRESSED_BYTES], P::D_U);
    let v = Poly::decompress(
        &ct_bytes[P::POLYVEC_COMPRESSED_BYTES..P::INDCPA_BYTES],
        P::D_V,
    );

    // Unpack secret key
    let skpv = PolyVec::<K>::frombytes(&sk_bytes[..P::INDCPA_SK_BYTES]);

    // m' = v − sᵀ · NTT⁻¹(NTT(u) · s_ntt)
    let mut b_ntt = b;
    b_ntt.ntt();

    let mut mp = Poly::zero();
    PolyVec::basemul_acc_montgomery(&mut mp, &skpv, &b_ntt);
    mp.invntt_tomont();

    let inner = mp;
    mp.sub(&v, &inner);
    mp.reduce();

    *m = mp.tomsg();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{MlKem512, MlKem768, MlKem1024};

    /// Manual pipeline test: build keygen/encrypt/decrypt step by step
    /// and check intermediate values.
    fn manual_pipeline<const K: usize>(eta1: usize, eta2: usize, _d_u: u32, _d_v: u32) {
        let coins = [42u8; SYMBYTES];

        // G(d ‖ k) → (ρ ‖ σ)
        let mut g_input = [0u8; SYMBYTES + 1];
        g_input[..SYMBYTES].copy_from_slice(&coins);
        g_input[SYMBYTES] = K as u8;
        let buf = hash::hash_g(&g_input);
        let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
        let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

        // Sample A (not transposed)
        let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut a, &public_seed, false);

        // Sample s and e
        let mut nonce = 0u8;
        let mut s = PolyVec::<K>::zero();
        for i in 0..K {
            s.polys[i] = Poly::getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        let mut e = PolyVec::<K>::zero();
        for i in 0..K {
            e.polys[i] = Poly::getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }

        // NTT(s), NTT(e) — reduce s before serialization
        s.ntt();
        s.reduce();
        e.ntt();

        // t = A*s + e (NTT domain)
        let mut t = PolyVec::<K>::zero();
        for i in 0..K {
            PolyVec::basemul_acc_montgomery(&mut t.polys[i], &a[i], &s);
            t.polys[i].tomont();
        }
        t.add_assign(&e);
        t.reduce();

        // Now test: encrypt zero message, decrypt, should get zero
        let enc_coins = [7u8; SYMBYTES];
        let zero_msg = [0u8; SYMBYTES];

        // Encryption
        let k_poly = Poly::frommsg(&zero_msg);

        // Sample Aᵀ
        let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut at, &public_seed, true);

        // Sample r, e1, e2
        let mut enc_nonce = 0u8;
        let mut r = PolyVec::<K>::zero();
        for i in 0..K {
            r.polys[i] = Poly::getnoise_eta(eta1, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let mut e1 = PolyVec::<K>::zero();
        for i in 0..K {
            e1.polys[i] = Poly::getnoise_eta(eta2, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let e2 = Poly::getnoise_eta(eta2, &enc_coins, enc_nonce);

        r.ntt();

        // u = Aᵀ·r + e1
        let mut u = PolyVec::<K>::zero();
        for i in 0..K {
            PolyVec::basemul_acc_montgomery(&mut u.polys[i], &at[i], &r);
        }
        // v = t·r + e2 + k
        let mut v = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut v, &t, &r);

        u.invntt_tomont();
        v.invntt_tomont();

        u.add_assign(&e1);
        v.add_assign(&e2);
        v.add_assign(&k_poly);

        u.reduce();
        v.reduce();

        // Now decrypt without compression (check exact result)
        // mp = s^T * NTT(u) (in NTT domain)
        let mut u_ntt = u.clone();
        u_ntt.ntt();

        let mut mp = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut mp, &s, &u_ntt);
        mp.invntt_tomont();

        // result = v - mp (should be ≈ k_poly + noise)
        let inner = mp;
        let mut result = Poly::zero();
        result.sub(&v, &inner);
        result.reduce();

        // For zero message, k_poly = 0, so result should be small noise
        let q = crate::params::Q as i32;
        let mut max_coeff = 0i32;
        for &c in result.coeffs.iter() {
            let cv = c as i32;
            let cv = if cv > q / 2 { cv - q } else if cv < -q / 2 { cv + q } else { cv };
            max_coeff = max_coeff.max(cv.abs());
        }
        eprintln!("max noise coefficient (no compression): {max_coeff}");
        assert!(
            max_coeff < q / 4,
            "noise too large: max_coeff={max_coeff} (limit={})",
            q / 4
        );

        // Now also check tomsg recovery
        let recovered = result.tomsg();
        assert_eq!(recovered, zero_msg, "zero-message recovery failed (no compression)");
    }

    #[test]
    fn manual_pipeline_k2() { manual_pipeline::<2>(3, 2, 10, 4); }
    #[test]
    fn manual_pipeline_k3() { manual_pipeline::<3>(2, 2, 10, 4); }
    #[test]
    fn manual_pipeline_k4() { manual_pipeline::<4>(2, 2, 11, 5); }

    /// Same as manual_pipeline but adds serialization/deserialization of pk/sk.
    fn serialized_manual_pipeline<const K: usize>(eta1: usize, eta2: usize) {
        let coins = [42u8; SYMBYTES];
        let polyvec_bytes = K * crate::params::POLYBYTES;
        let pk_bytes_len = polyvec_bytes + SYMBYTES;

        let mut g_input = [0u8; SYMBYTES + 1];
        g_input[..SYMBYTES].copy_from_slice(&coins);
        g_input[SYMBYTES] = K as u8;
        let buf = hash::hash_g(&g_input);
        let public_seed: [u8; SYMBYTES] = buf[..SYMBYTES].try_into().unwrap();
        let noise_seed: [u8; SYMBYTES] = buf[SYMBYTES..].try_into().unwrap();

        let mut a: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut a, &public_seed, false);

        let mut nonce = 0u8;
        let mut s = PolyVec::<K>::zero();
        for i in 0..K {
            s.polys[i] = Poly::getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        let mut e = PolyVec::<K>::zero();
        for i in 0..K {
            e.polys[i] = Poly::getnoise_eta(eta1, &noise_seed, nonce);
            nonce += 1;
        }
        s.ntt();
        s.reduce();
        e.ntt();

        let mut t = PolyVec::<K>::zero();
        for i in 0..K {
            PolyVec::basemul_acc_montgomery(&mut t.polys[i], &a[i], &s);
            t.polys[i].tomont();
        }
        t.add_assign(&e);
        t.reduce();

        // === Serialize pk and sk ===
        let mut pk_buf = vec![0u8; pk_bytes_len];
        t.tobytes(&mut pk_buf[..polyvec_bytes]);
        pk_buf[polyvec_bytes..].copy_from_slice(&public_seed);

        let mut sk_buf = vec![0u8; polyvec_bytes];
        s.tobytes(&mut sk_buf);

        // === Deserialize pk and sk ===
        let t2 = PolyVec::<K>::frombytes(&pk_buf[..polyvec_bytes]);
        let s2 = PolyVec::<K>::frombytes(&sk_buf);

        // Verify serialization roundtrip (values should match mod q)
        for k in 0..K {
            for j in 0..crate::params::N {
                let orig = crate::math::pack::csubq(t.polys[k].coeffs[j]);
                let deser = t2.polys[k].coeffs[j] as u16;
                assert_eq!(orig, deser, "t serialize mismatch at poly {k} coeff {j}: orig(reduced)={orig}, deser={deser}");
            }
        }

        // === Encryption using deserialized pk ===
        let enc_coins = [7u8; SYMBYTES];
        let zero_msg = [0u8; SYMBYTES];
        let k_poly = Poly::frommsg(&zero_msg);

        let mut at: [PolyVec<K>; K] = core::array::from_fn(|_| PolyVec::zero());
        gen_matrix::<K>(&mut at, &public_seed, true);

        let mut enc_nonce = 0u8;
        let mut r = PolyVec::<K>::zero();
        for i in 0..K {
            r.polys[i] = Poly::getnoise_eta(eta1, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let mut e1 = PolyVec::<K>::zero();
        for i in 0..K {
            e1.polys[i] = Poly::getnoise_eta(eta2, &enc_coins, enc_nonce);
            enc_nonce += 1;
        }
        let e2_poly = Poly::getnoise_eta(eta2, &enc_coins, enc_nonce);

        r.ntt();

        // Compare basemul results using direct vs deserialized t
        {
            let mut v_direct = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut v_direct, &t, &r);
            let mut v_deser = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut v_deser, &t2, &r);
            let q = crate::params::Q as i32;
            for j in 0..crate::params::N {
                let d = v_direct.coeffs[j] as i32;
                let s = v_deser.coeffs[j] as i32;
                let diff = ((d - s) % q + q) % q;
                let diff = diff.min(q - diff);
                assert_eq!(
                    diff, 0,
                    "basemul diff at coeff {j}: direct={d}, deser={s}, diff_mod_q={diff}"
                );
            }
        }

        let mut u = PolyVec::<K>::zero();
        for i in 0..K {
            PolyVec::basemul_acc_montgomery(&mut u.polys[i], &at[i], &r);
        }
        let mut v = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut v, &t2, &r); // Use deserialized t

        u.invntt_tomont();
        v.invntt_tomont();
        u.add_assign(&e1);
        v.add_assign(&e2_poly);
        v.add_assign(&k_poly);
        u.reduce();
        v.reduce();

        // === Decryption using deserialized sk ===
        let mut u_ntt = u.clone();
        u_ntt.ntt();

        // Verify basemul results match using direct vs deserialized s
        {
            let mut mp_direct = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut mp_direct, &s, &u_ntt);
            let mut mp_deser = Poly::zero();
            PolyVec::basemul_acc_montgomery(&mut mp_deser, &s2, &u_ntt);
            let q = crate::params::Q as i32;
            for j in 0..crate::params::N {
                let d = mp_direct.coeffs[j] as i32;
                let ds = mp_deser.coeffs[j] as i32;
                let diff = ((d - ds) % q + q) % q;
                let diff = diff.min(q - diff);
                assert_eq!(
                    diff, 0,
                    "sk basemul diff at coeff {j}: direct={d}, deser={ds}, diff_mod_q={diff}"
                );
            }
        }

        let mut mp = Poly::zero();
        PolyVec::basemul_acc_montgomery(&mut mp, &s2, &u_ntt); // Use deserialized s

        mp.invntt_tomont();

        let inner = mp;
        let mut result = Poly::zero();
        result.sub(&v, &inner);
        result.reduce();

        let recovered = result.tomsg();
        assert_eq!(recovered, zero_msg, "serialized pipeline: zero-message recovery failed");
    }

    #[test]
    fn serialized_manual_k2() { serialized_manual_pipeline::<2>(3, 2); }
    #[test]
    fn serialized_manual_k3() { serialized_manual_pipeline::<3>(2, 2); }

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
    fn indcpa_roundtrip_512() { indcpa_roundtrip::<MlKem512>(); }
    #[test]
    fn indcpa_roundtrip_768() { indcpa_roundtrip::<MlKem768>(); }
    #[test]
    fn indcpa_roundtrip_1024() { indcpa_roundtrip::<MlKem1024>(); }
}
