use core::simd::{Simd, prelude::*};

use crate::{Q, reduce::QINV};

/// Lane count used by kernel-level unit tests (16 × i16 = 256-bit).
#[cfg(test)]
const DEFAULT_LANES: usize = 16;

/// Barrett reduction: `r \equiv a \pmod{q}`, centered `|r| <= q/2`.
#[inline]
#[must_use]
pub(super) fn barrett_reduce_vec<const L: usize>(a: Simd<i16, L>) -> Simd<i16, L> {
    const V: i32 = 20159;
    let aw: Simd<i32, L> = a.cast();
    let t =
        ((Simd::<i32, L>::splat(V) * aw + Simd::splat(1 << 25)) >> Simd::splat(26)).cast::<i16>();
    a - t * Simd::splat(Q)
}

/// Montgomery reduction: `a * R^{-1} mod q`, `R = 2^{16}`.
#[inline]
#[must_use]
pub(super) fn montgomery_reduce_vec<const L: usize>(a: Simd<i32, L>) -> Simd<i16, L> {
    let qinv = Simd::<i32, L>::splat(QINV as i32);
    let q = Simd::<i32, L>::splat(Q as i32);
    let s16 = Simd::splat(16);
    let a_lo = (a << s16) >> s16;
    let t = ((a_lo * qinv) << s16) >> s16;
    ((a - t * q) >> s16).cast::<i16>()
}

/// Field multiply: `a * b * R^{-1} mod q`.
#[inline]
#[must_use]
pub(super) fn fqmul_vec<const L: usize>(a: Simd<i16, L>, b: Simd<i16, L>) -> Simd<i16, L> {
    montgomery_reduce_vec(a.cast::<i32>() * b.cast::<i32>())
}

#[cfg(test)]
mod tests {
    use core::simd::Simd;

    use super::{DEFAULT_LANES, fqmul_vec, montgomery_reduce_vec};
    use crate::{N, reduce};

    #[test]
    fn simd_montgomery_matches_scalar() {
        for val in [-100_000_i32, -1, 0, 1, 50_000, 100_000] {
            let v = Simd::<i32, DEFAULT_LANES>::splat(val);
            let result = montgomery_reduce_vec(v);
            let expected = reduce::montgomery_reduce(val);
            assert!(
                result.as_array().iter().all(|&x| x == expected),
                "mismatch for {val}: simd={}, scalar={expected}",
                result.as_array()[0]
            );
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
        let mut expected = [0i16; N];
        for i in 0..N {
            expected[i] = reduce::fqmul(a[i], b[i]);
        }
        let mut result = [0i16; N];
        for i in (0..N).step_by(DEFAULT_LANES) {
            let va = Simd::<i16, DEFAULT_LANES>::from_slice(&a[i..]);
            let vb = Simd::<i16, DEFAULT_LANES>::from_slice(&b[i..]);
            fqmul_vec(va, vb).copy_to_slice(&mut result[i..]);
        }
        assert_eq!(result, expected);
    }
}
