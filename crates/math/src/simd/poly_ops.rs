use core::simd::Simd;

use super::kernels::{barrett_reduce_vec, compress_d_vec, decompress_d_vec, fqmul_vec};
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
    const F: i16 = ((1u64 << 32) % (Q as u64)) as i16; // R^2 mod q = 1353
    let f = Simd::<i16, L>::splat(F);
    for chunk in c.as_chunks_mut::<L>().0 {
        *chunk = fqmul_vec(Simd::from_array(*chunk), f).into();
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

/// Fused inner product: `sum_{k=0}^{K-1} basemul(a[k], b[k])`, Barrett-reduced.
///
/// Keeps the accumulator in SOA (structure-of-arrays) form across all K
/// iterations, performing only ONE AOS↔SOA conversion pair per block instead
/// of K. This eliminates K-1 interleave/deinterleave passes, K-1 `poly_adds`,
/// and K zero-inits compared to the loop `basemul + add_assign`.
#[inline]
pub fn poly_inner_product<const K: usize>(a: [&[i16; N]; K], b: [&[i16; N]; K]) -> [i16; N] {
    // Cannot use dispatch_lanes! because we have two const generics (K, L).
    let mut result = match crate::simd::get_lane_width() {
        crate::simd::LaneWidth::W128Bit => poly_inner_product_lanes::<K, 8>(a, b),
        crate::simd::LaneWidth::W256Bit => poly_inner_product_lanes::<K, 16>(a, b),
        crate::simd::LaneWidth::W512Bit => poly_inner_product_lanes::<K, 32>(a, b),
        crate::simd::LaneWidth::W1024Bit => poly_inner_product_lanes::<K, 64>(a, b),
    };
    poly_reduce(&mut result);
    result
}

/// Process `L` blocks in parallel using SIMD de-interleave/interleave for the
/// stride-4 gather/scatter and `fqmul_vec` for the arithmetic.
#[inline]
fn poly_basemul_lanes<const L: usize>(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    let mut r = [0i16; N];
    for ((([a0, a1, a2, a3], [b0, b1, b2, b3]), z), out) in
        (a.as_chunks::<L>().0.as_chunks::<4>().0.iter())
            .zip(b.as_chunks::<L>().0.as_chunks::<4>().0.iter())
            .zip(crate::ntt::ZETAS[64..].as_chunks::<L>().0.iter())
            .zip(r.as_chunks_mut::<L>().0.as_chunks_mut::<4>().0.iter_mut())
    {
        // 4-way de-interleave (AOS→SOA): two passes of 2-way deinterleave
        // turn [a0,a1,a2,a3, b0,b1,b2,b3, ...] into four L-wide role vectors.
        let (t0, t1) = Simd::from_array(*a0).deinterleave(Simd::from_array(*a1));
        let (t2, t3) = Simd::from_array(*a2).deinterleave(Simd::from_array(*a3));
        let (a0, a2) = t0.deinterleave(t2);
        let (a1, a3) = t1.deinterleave(t3);

        let (t0, t1) = Simd::from_array(*b0).deinterleave(Simd::from_array(*b1));
        let (t2, t3) = Simd::from_array(*b2).deinterleave(Simd::from_array(*b3));
        let (b0, b2) = t0.deinterleave(t2);
        let (b1, b3) = t1.deinterleave(t3);

        let z = Simd::from_slice(z);

        let r0 = fqmul_vec(fqmul_vec(a1, b1), z) + fqmul_vec(a0, b0);
        let r1 = fqmul_vec(a0, b1) + fqmul_vec(a1, b0);

        let r2 = fqmul_vec(fqmul_vec(a3, b3), -z) + fqmul_vec(a2, b2);
        let r3 = fqmul_vec(a2, b3) + fqmul_vec(a3, b2);

        // 4-way re-interleave (SOA→AOS): reverse the deinterleave.
        let (lo02, hi02) = r0.interleave(r2);
        let (lo13, hi13) = r1.interleave(r3);
        let (out0, out1) = lo02.interleave(lo13);
        let (out2, out3) = hi02.interleave(hi13);

        *out = [out0, out1, out2, out3].map(Into::into);
    }
    r
}

/// Fused inner product kernel: block-outermost, K-innermost loop.
///
/// For each of the `N / (4 * L)` blocks, deinterleave both a[k] and b[k],
/// accumulate basemul products in SOA-format SIMD registers, and interleave
/// ONCE at the end. The K loop is const-generic so the compiler unrolls it.
#[inline]
fn poly_inner_product_lanes<const K: usize, const L: usize>(
    a: [&[i16; N]; K], b: [&[i16; N]; K],
) -> [i16; N] {
    let mut r = [0i16; N];
    let zetas = &crate::ntt::ZETAS[64..];

    let (r_chunks, _) = r.as_chunks_mut::<L>();
    for (bi, out) in (r_chunks.as_chunks_mut::<4>().0).iter_mut().enumerate() {
        let z = Simd::<i16, L>::from_slice(&zetas[bi * L..]);

        let mut acc0 = Simd::<i16, L>::splat(0);
        let mut acc1 = Simd::<i16, L>::splat(0);
        let mut acc2 = Simd::<i16, L>::splat(0);
        let mut acc3 = Simd::<i16, L>::splat(0);

        for k in 0..K {
            let (ak, _) = a[k].as_chunks::<L>();
            let (bk, _) = b[k].as_chunks::<L>();
            let ci = bi * 4; // base chunk index for this block

            // 4-way deinterleave (AOS→SOA) for a[k]
            let (t0, t1) = Simd::from_array(ak[ci]).deinterleave(Simd::from_array(ak[ci + 1]));
            let (t2, t3) = Simd::from_array(ak[ci + 2]).deinterleave(Simd::from_array(ak[ci + 3]));
            let (a0, a2) = t0.deinterleave(t2);
            let (a1, a3) = t1.deinterleave(t3);

            // 4-way deinterleave (AOS→SOA) for b[k]
            let (t0, t1) = Simd::from_array(bk[ci]).deinterleave(Simd::from_array(bk[ci + 1]));
            let (t2, t3) = Simd::from_array(bk[ci + 2]).deinterleave(Simd::from_array(bk[ci + 3]));
            let (b0, b2) = t0.deinterleave(t2);
            let (b1, b3) = t1.deinterleave(t3);

            // Basemul and accumulate in SOA form
            acc0 += fqmul_vec(fqmul_vec(a1, b1), z) + fqmul_vec(a0, b0);
            acc1 += fqmul_vec(a0, b1) + fqmul_vec(a1, b0);
            acc2 += fqmul_vec(fqmul_vec(a3, b3), -z) + fqmul_vec(a2, b2);
            acc3 += fqmul_vec(a2, b3) + fqmul_vec(a3, b2);
        }

        // 4-way re-interleave (SOA→AOS): only once per block
        let (lo02, hi02) = acc0.interleave(acc2);
        let (lo13, hi13) = acc1.interleave(acc3);
        let (out0, out1) = lo02.interleave(lo13);
        let (out2, out3) = hi02.interleave(hi13);

        *out = [out0, out1, out2, out3].map(Into::into);
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

    #[test]
    fn fused_inner_product_matches_basemul_loop() {
        // Build K=3 polynomial pairs with varied data.
        const K: usize = 3;
        let mut a_polys = [[0i16; N]; K];
        let mut b_polys = [[0i16; N]; K];
        for k in 0..K {
            for i in 0..N {
                a_polys[k][i] = ((k * N + i) as i16 * 13 + 7) % 1000 - 500;
                b_polys[k][i] = ((k * N + i) as i16 * 7 + 3) % 1000 - 300;
            }
        }

        // Reference: basemul + add + reduce (the old inner_product).
        let mut expected = poly_basemul(&a_polys[0], &b_polys[0]);
        for k in 1..K {
            let prod = poly_basemul(&a_polys[k], &b_polys[k]);
            poly_add_assign(&mut expected, &prod);
        }
        poly_reduce(&mut expected);

        // Fused version.
        let a_refs: [&[i16; N]; K] = [&a_polys[0], &a_polys[1], &a_polys[2]];
        let b_refs: [&[i16; N]; K] = [&b_polys[0], &b_polys[1], &b_polys[2]];
        let result = poly_inner_product(a_refs, b_refs);

        // Both should produce Barrett-reduced results that are congruent mod q.
        for i in 0..N {
            let e = reduce::barrett_reduce(expected[i]);
            let r = reduce::barrett_reduce(result[i]);
            assert_eq!(
                e, r,
                "mismatch at {i}: expected {e} (raw {}), got {r} (raw {})",
                expected[i], result[i]
            );
        }
    }
}
