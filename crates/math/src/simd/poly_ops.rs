use core::simd::{Simd, prelude::*};

use super::kernels::{barrett_reduce_vec, fqmul_vec, montgomery_reduce_vec};
use crate::{N, Q};

/// Barrett-reduce all `N` coefficients in-place.
#[inline]
pub fn poly_reduce(c: &mut [i16; N]) {
    super::dispatch_lanes!(poly_reduce_lanes(c));
}

#[inline]
fn poly_reduce_lanes<const L: usize>(c: &mut [i16; N]) {
    for ch in c.chunks_exact_mut(L) {
        barrett_reduce_vec(Simd::<i16, L>::from_slice(ch)).copy_to_slice(ch);
    }
}

/// `r[i] = a[i] + b[i]`.
#[inline]
pub fn poly_add(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    super::dispatch_lanes!(poly_add_lanes(r, a, b));
}

#[inline]
fn poly_add_lanes<const L: usize>(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&a[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va + vb).copy_to_slice(&mut r[i..]);
    }
}

/// `r[i] += b[i]`.
#[inline]
pub fn poly_add_assign(r: &mut [i16; N], b: &[i16; N]) {
    super::dispatch_lanes!(poly_add_assign_lanes(r, b));
}

#[inline]
fn poly_add_assign_lanes<const L: usize>(r: &mut [i16; N], b: &[i16; N]) {
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&r[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va + vb).copy_to_slice(&mut r[i..]);
    }
}

/// `r[i] = a[i] - b[i]`.
#[inline]
pub fn poly_sub(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    super::dispatch_lanes!(poly_sub_lanes(r, a, b));
}

#[inline]
fn poly_sub_lanes<const L: usize>(r: &mut [i16; N], a: &[i16; N], b: &[i16; N]) {
    for i in (0..N).step_by(L) {
        let va = Simd::<i16, L>::from_slice(&a[i..]);
        let vb = Simd::<i16, L>::from_slice(&b[i..]);
        (va - vb).copy_to_slice(&mut r[i..]);
    }
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
    for i in (0..N).step_by(L) {
        let v: Simd<i32, L> = Simd::<i16, L>::from_slice(&c[i..]).cast();
        montgomery_reduce_vec(v * f).copy_to_slice(&mut c[i..]);
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
    for i in (0..N).step_by(L) {
        let v = Simd::<i16, L>::from_slice(&c[i..]);
        fqmul_vec(v, s).copy_to_slice(&mut c[i..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce;

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
