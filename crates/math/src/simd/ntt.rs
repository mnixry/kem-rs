//! SIMD-accelerated NTT kernels.

use core::simd::{Simd, Swizzle};

use super::kernels::fqmul_vec;
use crate::N;

pub const Q64: i64 = crate::Q as i64;

#[cfg_attr(coverage_nightly, coverage(off))]
pub const fn pow_mod(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result: i64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

#[cfg_attr(coverage_nightly, coverage(off))]
const fn bitrev7(x: usize) -> usize {
    ((x >> 6) & 1)
        | (((x >> 5) & 1) << 1)
        | (((x >> 4) & 1) << 2)
        | (((x >> 3) & 1) << 3)
        | (((x >> 2) & 1) << 4)
        | (((x >> 1) & 1) << 5)
        | ((x & 1) << 6)
}

/// Centred representative of `val mod q` in `[-(q-1)/2, (q-1)/2]`.
#[cfg_attr(coverage_nightly, coverage(off))]
pub const fn centred(val: i64) -> i16 {
    if val > Q64 / 2 {
        (val - Q64) as i16
    } else {
        val as i16
    }
}

/// Twiddle factors in Montgomery form, from primitive 512th root \zeta = 17,
/// bit-reversed indexing.
///
/// `ZETAS[i] = \zeta^{BitRev_7(i)} * 2^{16}  (mod q)`, centred to signed.
///
/// Values match Appendix A of FIPS 203 (scaled into Montgomery domain).
pub const ZETAS: [i16; 128] = {
    const ZETA: i64 = 17;
    const MONT: i64 = 1 << 16;

    let mut zetas = [0i16; 128];
    let mut i = 0;
    while i < 128 {
        let val = pow_mod(ZETA, bitrev7(i) as i64, Q64) * MONT % Q64;
        zetas[i] = centred(val);
        i += 1;
    }
    zetas
};

mod swizzles {
    use core::{marker::PhantomData, simd::Swizzle};

    pub struct PackedDeinterleaveLo<const BLOCK: usize>(PhantomData<[usize; BLOCK]>);
    impl<const LANES: usize, const BLOCK: usize> Swizzle<LANES> for PackedDeinterleaveLo<BLOCK> {
        const INDEX: [usize; LANES] = const {
            let mut idx = [0usize; LANES];
            let mut dst = 0;
            let mut src = 0;
            while dst < LANES {
                let mut j = 0;
                while j < BLOCK {
                    idx[dst + j] = src + j;
                    j += 1;
                }
                dst += BLOCK;
                src += 2 * BLOCK;
            }
            idx
        };
    }

    pub struct PackedDeinterleaveHi<const BLOCK: usize>(PhantomData<[usize; BLOCK]>);
    impl<const LANES: usize, const BLOCK: usize> Swizzle<LANES> for PackedDeinterleaveHi<BLOCK> {
        const INDEX: [usize; LANES] = const {
            let mut idx = [0usize; LANES];
            let mut dst = 0;
            let mut src = BLOCK;
            while dst < LANES {
                let mut j = 0;
                while j < BLOCK {
                    idx[dst + j] = src + j;
                    j += 1;
                }
                dst += BLOCK;
                src += 2 * BLOCK;
            }
            idx
        };
    }

    pub struct PackedInterleaveFirst<const BLOCK: usize>(PhantomData<[usize; BLOCK]>);
    impl<const LANES: usize, const BLOCK: usize> Swizzle<LANES> for PackedInterleaveFirst<BLOCK> {
        const INDEX: [usize; LANES] = const {
            let groups_per_vec = LANES / (2 * BLOCK);
            let mut idx = [0usize; LANES];
            let mut dst = 0;
            let mut grp = 0;
            while grp < groups_per_vec {
                let mut j = 0;
                while j < BLOCK {
                    idx[dst + j] = grp * BLOCK + j; // from lo (first vector)
                    j += 1;
                }
                dst += BLOCK;
                j = 0;
                while j < BLOCK {
                    idx[dst + j] = LANES + grp * BLOCK + j; // from hi (second vector)
                    j += 1;
                }
                dst += BLOCK;
                grp += 1;
            }
            idx
        };
    }

    pub struct PackedInterleaveSecond<const BLOCK: usize>(PhantomData<[usize; BLOCK]>);
    impl<const LANES: usize, const BLOCK: usize> Swizzle<LANES> for PackedInterleaveSecond<BLOCK> {
        const INDEX: [usize; LANES] = const {
            let groups_per_vec = LANES / (2 * BLOCK);
            let mut idx = [0usize; LANES];
            let mut dst = 0;
            let mut grp = groups_per_vec;
            while grp < 2 * groups_per_vec {
                let mut j = 0;
                while j < BLOCK {
                    idx[dst + j] = grp * BLOCK + j; // from lo (first vector)
                    j += 1;
                }
                dst += BLOCK;
                j = 0;
                while j < BLOCK {
                    idx[dst + j] = LANES + grp * BLOCK + j; // from hi (second vector)
                    j += 1;
                }
                dst += BLOCK;
                grp += 1;
            }
            idx
        };
    }
}

/// Single NTT butterfly layer. `$sw` (SIMD width) must divide `$len`.
/// When `$sw == $len` the inner loop executes exactly once (no tail).
#[inline]
pub fn layer_fwd<const LEN: usize, const SW: usize>(r: &mut [i16; N], k: &mut usize) {
    for start in (0..N).step_by(2 * LEN) {
        let zeta = ZETAS[*k];
        *k += 1;
        let (lo, hi) = r[start..start + 2 * LEN].split_at_mut(LEN);
        let z = Simd::<i16, SW>::splat(zeta);
        for i in (0..LEN).step_by(SW) {
            let a = Simd::<i16, SW>::from_slice(&lo[i..]);
            let b = Simd::<i16, SW>::from_slice(&hi[i..]);
            let t = fqmul_vec(z, b);
            (a + t).copy_to_slice(&mut lo[i..i + SW]);
            (a - t).copy_to_slice(&mut hi[i..i + SW]);
        }
    }
}

// Inverse butterfly WITHOUT Barrett reduction on the sum.
// Caller must ensure |a + b| and |b - a| fit in i16 (no wrapping).
#[inline]
pub fn layer_inv_nored<const LEN: usize, const SW: usize>(r: &mut [i16; N], k: &mut usize) {
    for start in (0..N).step_by(2 * LEN) {
        let zeta = ZETAS[*k];
        *k = k.wrapping_sub(1);
        let (lo, hi) = r[start..start + 2 * LEN].split_at_mut(LEN);
        let z = Simd::<i16, SW>::splat(zeta);
        for i in (0..LEN).step_by(SW) {
            let a = Simd::<i16, SW>::from_slice(&lo[i..]);
            let b = Simd::<i16, SW>::from_slice(&hi[i..]);
            (a + b).copy_to_slice(&mut lo[i..i + SW]);
            fqmul_vec(z, b - a).copy_to_slice(&mut hi[i..i + SW]);
        }
    }
}

#[inline]
pub fn layer_fwd_packed<const LEN: usize, const LANES: usize>(r: &mut [i16; N], k: &mut usize) {
    let groups = LANES / LEN;
    for start in (0..N).step_by(2 * LANES) {
        let v0 = Simd::<i16, LANES>::from_slice(&r[start..]);
        let v1 = Simd::<i16, LANES>::from_slice(&r[start + LANES..]);

        let lo = swizzles::PackedDeinterleaveLo::<LEN>::concat_swizzle(v0, v1);
        let hi = swizzles::PackedDeinterleaveHi::<LEN>::concat_swizzle(v0, v1);

        // Construct zeta vector: each zeta repeated $len times.
        let mut z_arr = [0i16; LANES];
        for g in 0..groups {
            let zeta = ZETAS[k.wrapping_add(g)];
            for j in 0..LEN {
                z_arr[g * LEN + j] = zeta;
            }
        }
        *k += groups;
        let z = Simd::<i16, LANES>::from_array(z_arr);

        let t = fqmul_vec(z, hi);
        let new_lo = lo + t;
        let new_hi = lo - t;

        let r0: Simd<i16, LANES> =
            swizzles::PackedInterleaveFirst::<LEN>::concat_swizzle(new_lo, new_hi);
        let r1: Simd<i16, LANES> =
            swizzles::PackedInterleaveSecond::<LEN>::concat_swizzle(new_lo, new_hi);
        r0.copy_to_slice(&mut r[start..start + LANES]);
        r1.copy_to_slice(&mut r[start + LANES..start + 2 * LANES]);
    }
}

#[inline]
pub fn layer_inv_nored_packed<const LEN: usize, const LANES: usize>(
    r: &mut [i16; N], k: &mut usize,
) {
    let groups = LANES / LEN;
    for start in (0..N).step_by(2 * LANES) {
        let v0 = Simd::<i16, LANES>::from_slice(&r[start..]);
        let v1 = Simd::<i16, LANES>::from_slice(&r[start + LANES..]);

        let a = swizzles::PackedDeinterleaveLo::<LEN>::concat_swizzle(v0, v1);
        let b = swizzles::PackedDeinterleaveHi::<LEN>::concat_swizzle(v0, v1);

        // Construct zeta vector: k decrements per group.
        let mut z_arr = [0i16; LANES];
        for g in 0..groups {
            let zeta = ZETAS[k.wrapping_sub(g)];
            for j in 0..LEN {
                z_arr[g * LEN + j] = zeta;
            }
        }
        *k = k.wrapping_sub(groups);
        let z = Simd::<i16, LANES>::from_array(z_arr);

        let new_lo = a + b;
        let new_hi = fqmul_vec(z, b - a);

        let r0: Simd<i16, LANES> =
            swizzles::PackedInterleaveFirst::<LEN>::concat_swizzle(new_lo, new_hi);
        let r1: Simd<i16, LANES> =
            swizzles::PackedInterleaveSecond::<LEN>::concat_swizzle(new_lo, new_hi);
        r0.copy_to_slice(&mut r[start..start + LANES]);
        r1.copy_to_slice(&mut r[start + LANES..start + 2 * LANES]);
    }
}
