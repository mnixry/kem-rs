//! Portable SIMD kernels for ML-KEM polynomial arithmetic.
//!
//! All operations are lane-generic over `L` (must satisfy `SupportedLaneCount`).
//! `N = 256` divides evenly by every supported lane count, so no scalar tail
//! is needed for full-polynomial passes.
//!
//! Default lane count: `DEFAULT_LANES = 16` (256-bit i16 vectors).
//! Use `_lanes::<L>` variants to override.
//!
//! Reference: PQClean ml-kem-768/avx2 -- `vpmullw`/`vpmulhw` for Montgomery
//! multiply, `vpaddw`/`vpsubw` for NTT butterfly add/sub.

use std::simd::prelude::*;
use std::simd::Simd;

use crate::math::reduce::QINV;
use crate::params::{N, Q};

/// 16 x i16 = 256-bit default. Matches SSE4 / NEON register width.
pub const DEFAULT_LANES: usize = 16;

// -- element-wise SIMD kernels -----------------------------------------------

/// Barrett reduction: `r \equiv a \pmod{q}`, centered `|r| <= q/2`.
#[inline]
pub fn barrett_reduce_vec<const L: usize>(a: Simd<i16, L>) -> Simd<i16, L>
{
    // t = round(V * a / 2^{26}), V = round(2^{26} / q) = 20159
    const V: i32 = 20159;
    let aw: Simd<i32, L> = a.cast();
    let t = ((Simd::<i32, L>::splat(V) * aw + Simd::splat(1 << 25)) >> Simd::splat(26))
        .cast::<i16>();
    a - t * Simd::splat(Q)
}

/// Montgomery reduction: `a * R^{-1} mod q`, `R = 2^{16}`.
#[inline]
pub fn montgomery_reduce_vec<const L: usize>(a: Simd<i32, L>) -> Simd<i16, L>
{
    let qinv = Simd::<i32, L>::splat(QINV as i32);
    let q = Simd::<i32, L>::splat(Q as i32);
    let s16 = Simd::splat(16);
    // t = (a as i16).wrapping_mul(QINV), sign-extended to i32
    let a_lo = (a << s16) >> s16;
    let t = ((a_lo * qinv) << s16) >> s16;
    ((a - t * q) >> s16).cast::<i16>()
}

/// Field multiply: `a * b * R^{-1} mod q`.
#[inline]
pub fn fqmul_vec<const L: usize>(a: Simd<i16, L>, b: Simd<i16, L>) -> Simd<i16, L>
{
    montgomery_reduce_vec(a.cast::<i32>() * b.cast::<i32>())
}

// -- polynomial-level passes -------------------------------------------------

/// Barrett-reduce all `N` coefficients in-place.
#[inline]
pub fn poly_reduce(c: &mut [i16; N]) {
    poly_reduce_lanes::<DEFAULT_LANES>(c);
}

#[inline]
pub fn poly_reduce_lanes<const L: usize>(c: &mut [i16; N])
{
    for ch in c.chunks_exact_mut(L) {
        barrett_reduce_vec(Simd::<i16, L>::from_slice(ch)).copy_to_slice(ch);
    }
}

/// `r[i] = a[i] + b[i]`.
#[inline]
pub fn poly_add(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    poly_add_lanes::<DEFAULT_LANES>(r, a, b);
}

#[inline]
pub fn poly_add_lanes<const L: usize>(r: &mut [i16; N], a: &[i16; N], b: &[i16; N])
{
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&a[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va + vb).copy_to_slice(&mut r[i..]);
    }
}

/// `r[i] += b[i]`.
#[inline]
pub fn poly_add_assign(r: &mut [i16; N], b: &[i16; N]) {
    poly_add_assign_lanes::<DEFAULT_LANES>(r, b);
}

#[inline]
pub fn poly_add_assign_lanes<const L: usize>(r: &mut [i16; N], b: &[i16; N])
{
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&r[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va + vb).copy_to_slice(&mut r[i..]);
    }
}

/// `r[i] = a[i] - b[i]`.
#[inline]
pub fn poly_sub(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    poly_sub_lanes::<DEFAULT_LANES>(r, a, b);
}

#[inline]
pub fn poly_sub_lanes<const L: usize>(r: &mut [i16; N], a: &[i16; N], b: &[i16; N])
{
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&a[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va - vb).copy_to_slice(&mut r[i..]);
    }
}

/// Convert all coefficients to Montgomery domain: `c_i <- c_i * R mod q`.
#[inline]
pub fn poly_tomont(c: &mut [i16; N]) {
    poly_tomont_lanes::<DEFAULT_LANES>(c);
}

#[inline]
pub fn poly_tomont_lanes<const L: usize>(c: &mut [i16; N])
{
    const F: i32 = ((1u64 << 32) % (Q as u64)) as i32; // R^2 mod q = 1353
    let f = Simd::<i32, L>::splat(F);
    for i in (0..N).step_by(L) {
        let v: Simd<i32, L> = Simd::<i16, L>::from_slice(&c[i..]).cast();
        montgomery_reduce_vec(v * f).copy_to_slice(&mut c[i..]);
    }
}

/// `c_i <- c_i * scalar * R^{-1} mod q`.
#[inline]
pub fn poly_fqmul_scalar(c: &mut [i16; N], scalar: i16) {
    poly_fqmul_scalar_lanes::<DEFAULT_LANES>(c, scalar);
}

#[inline]
pub fn poly_fqmul_scalar_lanes<const L: usize>(c: &mut [i16; N], scalar: i16)
{
    let s = Simd::<i16, L>::splat(scalar);
    for i in (0..N).step_by(L) {
        let v = Simd::<i16, L>::from_slice(&c[i..]);
        fqmul_vec(v, s).copy_to_slice(&mut c[i..]);
    }
}

// -- NTT butterfly -----------------------------------------------------------

/// Forward butterfly: `(lo, hi) <- (lo + t, lo - t)` where `t = zeta * hi * R^{-1}`.
///
/// SIMD for chunks of `DEFAULT_LANES`, scalar fallback for the remainder.
#[inline]
pub fn butterfly_fwd(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
    butterfly_fwd_lanes::<DEFAULT_LANES>(lo, hi, zeta);
}

#[inline]
pub fn butterfly_fwd_lanes<const L: usize>(lo: &mut [i16], hi: &mut [i16], zeta: i16)
{
    debug_assert_eq!(lo.len(), hi.len());
    let n = lo.len();
    let z = Simd::<i16, L>::splat(zeta);
    let simd_end = n - (n % L);

    for i in (0..simd_end).step_by(L) {
        let a = Simd::<i16, L>::from_slice(&lo[i..]);
        let b = Simd::<i16, L>::from_slice(&hi[i..]);
        let t = fqmul_vec(z, b);
        (a + t).copy_to_slice(&mut lo[i..]);
        (a - t).copy_to_slice(&mut hi[i..]);
    }
    for i in simd_end..n {
        let t = crate::math::reduce::fqmul(zeta, hi[i]);
        hi[i] = lo[i] - t;
        lo[i] += t;
    }
}

/// Inverse butterfly: `(lo, hi) <- (barrett(lo+hi), zeta*(hi-lo)*R^{-1})`.
#[inline]
pub fn butterfly_inv(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
    butterfly_inv_lanes::<DEFAULT_LANES>(lo, hi, zeta);
}

#[inline]
pub fn butterfly_inv_lanes<const L: usize>(lo: &mut [i16], hi: &mut [i16], zeta: i16)
{
    debug_assert_eq!(lo.len(), hi.len());
    let n = lo.len();
    let z = Simd::<i16, L>::splat(zeta);
    let simd_end = n - (n % L);

    for i in (0..simd_end).step_by(L) {
        let a = Simd::<i16, L>::from_slice(&lo[i..]);
        let b = Simd::<i16, L>::from_slice(&hi[i..]);
        barrett_reduce_vec(a + b).copy_to_slice(&mut lo[i..]);
        fqmul_vec(z, b - a).copy_to_slice(&mut hi[i..]);
    }
    for i in simd_end..n {
        let t = lo[i];
        lo[i] = crate::math::reduce::barrett_reduce(t + hi[i]);
        hi[i] = crate::math::reduce::fqmul(zeta, hi[i] - t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::reduce;

    #[test]
    fn simd_barrett_matches_scalar() {
        let mut data = [0i16; N];
        for (i, c) in data.iter_mut().enumerate() {
            *c = (i as i16 * 37) - 1000;
        }
        let expected: Vec<i16> = data.iter().map(|&c| reduce::barrett_reduce(c)).collect();
        poly_reduce(&mut data);
        assert_eq!(&data[..], &expected[..]);
    }

    #[test]
    fn simd_montgomery_matches_scalar() {
        for val in [-100000i32, -1, 0, 1, 50000, 100000] {
            let v = Simd::<i32, DEFAULT_LANES>::splat(val);
            let result = montgomery_reduce_vec(v);
            let expected = reduce::montgomery_reduce(val);
            assert!(result.as_array().iter().all(|&x| x == expected),
                "mismatch for {val}: simd={}, scalar={expected}", result.as_array()[0]);
        }
    }

    #[test]
    fn simd_fqmul_matches_scalar() {
        let mut a = [0i16; N];
        let mut b = [0i16; N];
        for i in 0..N {
            a[i] = (i as i16 * 13) - 500;
            b[i] = (i as i16 * 7) + 100;
        }
        let expected: Vec<i16> = a.iter().zip(b.iter())
            .map(|(&x, &y)| reduce::fqmul(x, y)).collect();
        let mut result = [0i16; N];
        for i in (0..N).step_by(DEFAULT_LANES) {
            let va = Simd::<i16, DEFAULT_LANES>::from_slice(&a[i..]);
            let vb = Simd::<i16, DEFAULT_LANES>::from_slice(&b[i..]);
            fqmul_vec(va, vb).copy_to_slice(&mut result[i..]);
        }
        assert_eq!(&result[..], &expected[..]);
    }

    #[test]
    fn simd_butterfly_fwd_matches_scalar() {
        let n = 32;
        let mut lo_s = vec![0i16; n];
        let mut hi_s = vec![0i16; n];
        for i in 0..n {
            lo_s[i] = (i as i16) * 13;
            hi_s[i] = (i as i16) * 7 + 100;
        }
        let mut lo_v = lo_s.clone();
        let mut hi_v = hi_s.clone();
        let zeta = 1234i16;

        butterfly_fwd(&mut lo_v, &mut hi_v, zeta);
        for i in 0..n {
            let t = reduce::fqmul(zeta, hi_s[i]);
            hi_s[i] = lo_s[i] - t;
            lo_s[i] += t;
        }
        assert_eq!(lo_v, lo_s);
        assert_eq!(hi_v, hi_s);
    }

    #[test]
    fn simd_butterfly_inv_matches_scalar() {
        let n = 32;
        let mut lo_s = vec![0i16; n];
        let mut hi_s = vec![0i16; n];
        for i in 0..n {
            lo_s[i] = (i as i16) * 11 - 200;
            hi_s[i] = (i as i16) * 5 + 50;
        }
        let mut lo_v = lo_s.clone();
        let mut hi_v = hi_s.clone();
        let zeta = -567i16;

        butterfly_inv(&mut lo_v, &mut hi_v, zeta);
        for i in 0..n {
            let t = lo_s[i];
            lo_s[i] = reduce::barrett_reduce(t + hi_s[i]);
            hi_s[i] = reduce::fqmul(zeta, hi_s[i] - t);
        }
        assert_eq!(lo_v, lo_s);
        assert_eq!(hi_v, hi_s);
    }
}
