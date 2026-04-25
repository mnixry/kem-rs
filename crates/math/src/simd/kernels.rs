use core::simd::{Simd, prelude::*};

use crate::{Q, reduce::QINV};

/// Multiply high: `(a * b) >> 16` (`vpmulhw`).
macro_rules! mulhi {
    ($a:expr, $b:expr) => {
        ((($a).cast::<i32>() * ($b).cast::<i32>()) >> Simd::splat(16)).cast::<i16>()
    };
}

/// Barrett reduction: `r \equiv a \pmod{q}`, centered `|r| <= q/2`.
#[inline(always)]
#[must_use]
pub fn barrett_reduce_vec<const L: usize>(a: Simd<i16, L>) -> Simd<i16, L> {
    const V: i16 = 20159;
    let t = mulhi!(a, Simd::splat(V)); // (a * V) >> 16 -> vpmulhw
    let t = (t + Simd::splat(1 << 9)) >> Simd::splat(10); // remaining shift + rounding
    a - t * Simd::splat(Q) // -> vpmullw, vpsubw
}

/// Field multiply: `a * b * R^{-1} mod q`.
#[inline(always)]
#[must_use]
pub fn fqmul_vec<const L: usize>(a: Simd<i16, L>, b: Simd<i16, L>) -> Simd<i16, L> {
    let ab_lo = a * b; // wrapping i16 mul -> vpmullw
    let ab_hi = mulhi!(a, b); // high i16 half   -> vpmulhw
    let t = ab_lo * Simd::splat(QINV); // wrapping i16 mul -> vpmullw
    let tq_hi = mulhi!(t, Simd::splat(Q)); // high i16 half   -> vpmulhw
    ab_hi - tq_hi //                 -> vpsubw
}

/// ceil(2^40 / Q). Exact for all numerators < 2^23 (max compress numerator is
/// ~2^22.7).
///
/// Proof of exactness: with M = ceil(2^S/Q) and n < 2^23, the Barrett estimate
/// n*M/2^S ∈ \[n/Q, n/Q + 2^{-17}). Since max frac(n/Q) = (Q-1)/Q ≈ 0.9997 < 1
/// - 2^{-17}, the floor never overshoots.
const COMPRESS_BARRETT_M: i64 = {
    let q = Q as u64;
    (1u64 << 40).div_ceil(q)
} as i64;

/// Compress: `floor((csubq(a) << d + Q/2) / Q) & ((1 << d) - 1)`.
///
/// Replaces scalar `u32` division with Barrett reciprocal multiplication in
/// `i64`. Input: `i16` coefficients in `[-Q, Q-1]`. Output: compressed `i16` in
/// `[0, 2^d - 1]`.
#[inline(always)]
#[must_use]
pub fn compress_d_vec<const L: usize>(a: Simd<i16, L>, d: u32) -> Simd<i16, L> {
    let q = Simd::<i32, L>::splat(Q as i32);
    let a32: Simd<i32, L> = a.cast();

    // csubq: add Q when coefficient is negative → [0, Q-1]
    let neg_mask = a32 >> Simd::splat(31);
    let x = a32 + (neg_mask & q);

    // numerator = (x << d) + floor(Q/2)
    let num = (x << Simd::splat(d as i32)) + Simd::splat(Q as i32 / 2);

    // Barrett: floor(num / Q) = (num * M) >> 40
    let num64: Simd<i64, L> = num.cast();
    let quot = (num64 * Simd::<i64, L>::splat(COMPRESS_BARRETT_M)) >> Simd::splat(40i64);

    (quot.cast::<i32>() & Simd::splat((1i32 << d) - 1)).cast::<i16>()
}

/// Decompress: `(y * Q + 2^(d-1)) >> d`.
///
/// Input: compressed `i16` in `[0, 2^d - 1]`. Output: decompressed `i16`
/// coefficient.
#[inline(always)]
#[must_use]
pub fn decompress_d_vec<const L: usize>(y: Simd<i16, L>, d: u32) -> Simd<i16, L> {
    let y32: Simd<i32, L> = y.cast();
    let result =
        (y32 * Simd::splat(Q as i32) + Simd::splat(1i32 << (d - 1))) >> Simd::splat(d as i32);
    result.cast::<i16>()
}

#[cfg(test)]
mod tests {
    use core::simd::Simd;

    use super::{compress_d_vec, decompress_d_vec, fqmul_vec};
    use crate::{N, Q, compress::csubq, reduce};

    const DEFAULT_LANES: usize = 16;

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

    fn scalar_compress(x: u16, d: u32) -> u16 {
        ((((x as u32) << d).wrapping_add((Q as u32) / 2) / (Q as u32)) & ((1u32 << d) - 1)) as u16
    }

    fn scalar_decompress(y: u16, d: u32) -> u16 {
        (((y as u32) * (Q as u32) + (1u32 << (d - 1))) >> d) as u16
    }

    #[test]
    fn barrett_compress_exact_for_all_inputs() {
        for d in [1u32, 4, 5, 10, 11] {
            for x_raw in 0..Q {
                let x_pos = csubq(x_raw) as i16;
                let expected = scalar_compress(x_pos as u16, d);

                let v = Simd::<i16, DEFAULT_LANES>::splat(x_raw);
                let result = compress_d_vec(v, d);
                assert!(
                    result.as_array().iter().all(|&r| r as u16 == expected),
                    "compress mismatch d={d} x={x_raw}: simd={}, scalar={expected}",
                    result.as_array()[0]
                );
            }
            // also verify negative inputs (csubq path)
            for x_neg in (-(Q - 1))..0 {
                let x_pos = csubq(x_neg);
                let expected = scalar_compress(x_pos, d);

                let v = Simd::<i16, DEFAULT_LANES>::splat(x_neg);
                let result = compress_d_vec(v, d);
                assert!(
                    result.as_array().iter().all(|&r| r as u16 == expected),
                    "compress mismatch d={d} x={x_neg}: simd={}, scalar={expected}",
                    result.as_array()[0]
                );
            }
        }
    }

    #[test]
    fn simd_decompress_matches_scalar() {
        for d in [1u32, 4, 5, 10, 11] {
            let max_val = (1u16 << d) - 1;
            for y in 0..=max_val {
                let expected = scalar_decompress(y, d);

                let v = Simd::<i16, DEFAULT_LANES>::splat(y as i16);
                let result = decompress_d_vec(v, d);
                assert!(
                    result.as_array().iter().all(|&r| r as u16 == expected),
                    "decompress mismatch d={d} y={y}: simd={}, scalar={expected}",
                    result.as_array()[0]
                );
            }
        }
    }
}
