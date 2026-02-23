use core::simd::{Simd, prelude::*};

use super::kernels::{
    barrett_reduce_vec, compress_d_vec, decompress_d_vec, fqmul_vec, montgomery_reduce_vec,
};
use crate::{N, Q};

/// Barrett-reduce all `N` coefficients in-place.
#[inline]
pub fn poly_reduce(c: &mut [i16; N]) {
    super::dispatch_lanes!(poly_reduce_lanes(c));
}

#[inline]
fn poly_reduce_lanes<const L: usize>(c: &mut [i16; N]) {
    for ch in c.as_chunks_mut::<L>().0 {
        *ch = barrett_reduce_vec(Simd::from_array(*ch)).into();
    }
}

/// `r[i] = a[i] + b[i]`.
#[inline]
pub fn poly_add(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    super::dispatch_lanes!(poly_add_lanes(a, b))
}

#[inline]
fn poly_add_lanes<const L: usize>(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    let mut ret = [0i16; N];
    for ((a, b), r) in (a.as_chunks::<L>().0)
        .iter()
        .zip(b.as_chunks::<L>().0.iter())
        .zip(ret.as_chunks_mut::<L>().0.iter_mut())
    {
        let va = Simd::from_array(*a);
        let vb = Simd::from_array(*b);
        *r = (va + vb).into();
    }
    ret
}

/// `r[i] += b[i]`.
#[inline]
pub fn poly_add_assign(r: &mut [i16; N], b: &[i16; N]) {
    super::dispatch_lanes!(poly_add_assign_lanes(r, b));
}

#[inline]
fn poly_add_assign_lanes<const L: usize>(r: &mut [i16; N], b: &[i16; N]) {
    for (ret, b) in (r.as_chunks_mut::<L>().0)
        .iter_mut()
        .zip(b.as_chunks::<L>().0)
    {
        let mut r = Simd::from_array(*ret);
        r += Simd::from_array(*b);
        *ret = r.into();
    }
}

/// `r[i] = a[i] - b[i]`.
#[inline]
pub fn poly_sub(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    super::dispatch_lanes!(poly_sub_lanes(a, b))
}

#[inline]
fn poly_sub_lanes<const L: usize>(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    let mut ret = [0i16; N];
    for ((a, b), r) in (a.as_chunks::<L>().0)
        .iter()
        .zip(b.as_chunks::<L>().0.iter())
        .zip(ret.as_chunks_mut::<L>().0.iter_mut())
    {
        let va = Simd::from_array(*a);
        let vb = Simd::from_array(*b);
        *r = (va - vb).into();
    }
    ret
}

/// Convert all coefficients to Montgomery domain: `c_i <- c_i * R mod q`.
#[inline]
pub fn poly_to_montgomery(c: &mut [i16; N]) {
    super::dispatch_lanes!(poly_to_montgomery_lanes(c));
}

#[inline]
fn poly_to_montgomery_lanes<const L: usize>(c: &mut [i16; N]) {
    const F: i32 = ((1u64 << 32) % (Q as u64)) as i32; // R^2 mod q = 1353
    let f = Simd::<i32, L>::splat(F);
    for chunk in c.as_chunks_mut::<L>().0 {
        *chunk = montgomery_reduce_vec(Simd::from_array(*chunk).cast() * f).into();
    }
}

/// `c_i <- c_i * scalar * R^{-1} mod q`.
#[inline]
pub fn poly_mul_scalar_montgomery(c: &mut [i16; N], scalar: i16) {
    super::dispatch_lanes!(poly_mul_scalar_montgomery_lanes(c, scalar));
}

#[inline]
fn poly_mul_scalar_montgomery_lanes<const L: usize>(c: &mut [i16; N], scalar: i16) {
    let s = Simd::<i16, L>::splat(scalar);
    for chunk in c.as_chunks_mut::<L>().0 {
        *chunk = fqmul_vec(Simd::from_array(*chunk), s).into();
    }
}

/// SIMD csubq + Barrett compress all `N` coefficients.
///
/// `out[i] = compress(coeffs[i], d)` for `d ∈ {1,4,5,10,11}`.
#[inline]
pub fn poly_compress_coeffs(coeffs: &[i16; N], d: u32) -> [i16; N] {
    super::dispatch_lanes!(poly_compress_coeffs_lanes(coeffs, d))
}

#[inline]
fn poly_compress_coeffs_lanes<const L: usize>(coeffs: &[i16; N], d: u32) -> [i16; N] {
    const { assert!(N.is_multiple_of(L)) }
    let mut outs = [0i16; N];
    for (out, chunk) in (outs.as_chunks_mut::<L>().0)
        .iter_mut()
        .zip(coeffs.as_chunks::<L>().0)
    {
        let a = Simd::from_array(*chunk);
        *out = compress_d_vec(a, d).into();
    }
    outs
}

/// SIMD decompress all `N` coefficients.
///
/// `out[i] = decompress(compressed[i], d)` for `d ∈ {1,4,5,10,11}`.
#[inline]
pub fn poly_decompress_coeffs(compressed: &[i16; N], d: u32) -> [i16; N] {
    super::dispatch_lanes!(poly_decompress_coeffs_lanes(compressed, d))
}

#[inline]
fn poly_decompress_coeffs_lanes<const L: usize>(compressed: &[i16; N], d: u32) -> [i16; N] {
    const { assert!(N.is_multiple_of(L)) }
    let mut outs = [0i16; N];
    for (out, chunk) in (outs.as_chunks_mut::<L>().0)
        .iter_mut()
        .zip(compressed.as_chunks::<L>().0)
    {
        let y = Simd::from_array(*chunk);
        *out = decompress_d_vec(y, d).into();
    }
    outs
}

/// Pointwise basemul: 64 degree-1 block multiplications in NTT domain.
#[inline]
pub fn poly_basemul(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    super::dispatch_lanes!(poly_basemul_lanes(a, b))
}

/// Process `L` blocks in parallel using SIMD de-interleave/interleave for the
/// stride-4 gather/scatter and `fqmul_vec` for the arithmetic.
#[inline]
fn poly_basemul_lanes<const L: usize>(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    let mut r = [0i16; N];
    for blk in (0..N / 4).step_by(L) {
        let base = 4 * blk;

        // 4-way de-interleave (AOS→SOA): two passes of 2-way deinterleave
        // turn [a0,a1,a2,a3, b0,b1,b2,b3, ...] into four L-wide role vectors.
        let (t0, t1) =
            Simd::<i16, L>::from_slice(&a[base..]).deinterleave(Simd::from_slice(&a[base + L..]));
        let (t2, t3) = Simd::<i16, L>::from_slice(&a[base + 2 * L..])
            .deinterleave(Simd::from_slice(&a[base + 3 * L..]));
        let (a0, a2) = t0.deinterleave(t2);
        let (a1, a3) = t1.deinterleave(t3);

        let (t0, t1) =
            Simd::<i16, L>::from_slice(&b[base..]).deinterleave(Simd::from_slice(&b[base + L..]));
        let (t2, t3) = Simd::<i16, L>::from_slice(&b[base + 2 * L..])
            .deinterleave(Simd::from_slice(&b[base + 3 * L..]));
        let (b0, b2) = t0.deinterleave(t2);
        let (b1, b3) = t1.deinterleave(t3);

        let z = Simd::<i16, L>::from_slice(&crate::ntt::ZETAS[64 + blk..]);

        let r0 = fqmul_vec(fqmul_vec(a1, b1), z) + fqmul_vec(a0, b0);
        let r1 = fqmul_vec(a0, b1) + fqmul_vec(a1, b0);

        let r2 = fqmul_vec(fqmul_vec(a3, b3), -z) + fqmul_vec(a2, b2);
        let r3 = fqmul_vec(a2, b3) + fqmul_vec(a3, b2);

        // 4-way re-interleave (SOA→AOS): reverse the deinterleave.
        let (lo02, hi02) = r0.interleave(r2);
        let (lo13, hi13) = r1.interleave(r3);
        let (out0, out1) = lo02.interleave(lo13);
        let (out2, out3) = hi02.interleave(hi13);

        for (i, out) in [out0, out1, out2, out3].iter().enumerate() {
            out.copy_to_slice(&mut r[base + i * L..]);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce;

    #[test]
    fn simd_basemul_matches_scalar() {
        use crate::ntt::ZETAS;

        let mut a = [0i16; N];
        let mut b = [0i16; N];
        for i in 0..N {
            a[i] = (i as i16 * 13) - 500;
            b[i] = (i as i16 * 7) + 100;
        }

        let mut expected = [0i16; N];
        for i in 0..N / 4 {
            let base = 4 * i;
            let zeta = ZETAS[64 + i];
            expected[base] = reduce::fqmul(reduce::fqmul(a[base + 1], b[base + 1]), zeta);
            expected[base] += reduce::fqmul(a[base], b[base]);
            expected[base + 1] = reduce::fqmul(a[base], b[base + 1]);
            expected[base + 1] += reduce::fqmul(a[base + 1], b[base]);
            expected[base + 2] = reduce::fqmul(reduce::fqmul(a[base + 3], b[base + 3]), -zeta);
            expected[base + 2] += reduce::fqmul(a[base + 2], b[base + 2]);
            expected[base + 3] = reduce::fqmul(a[base + 2], b[base + 3]);
            expected[base + 3] += reduce::fqmul(a[base + 3], b[base + 2]);
        }

        let result = poly_basemul(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn simd_barrett_matches_scalar() {
        let mut data = [0i16; N];
        for (i, c) in data.iter_mut().enumerate() {
            *c = (i as i16 * 37) - 1000;
        }
        let mut expected = [0i16; N];
        for (e, &c) in expected.iter_mut().zip(data.iter()) {
            *e = reduce::barrett_reduce(c);
        }
        poly_reduce(&mut data);
        assert_eq!(data, expected);
    }
}
