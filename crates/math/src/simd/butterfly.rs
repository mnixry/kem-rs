use core::simd::Simd;

use super::kernels::{barrett_reduce_vec, fqmul_vec};

/// Forward butterfly: `(lo, hi) <- (lo + t, lo - t)` where `t = zeta * hi *
/// R^{-1}`.
#[inline]
pub fn butterfly_forward(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
    super::dispatch_lanes!(butterfly_forward_lanes(lo, hi, zeta));
}

#[inline]
fn butterfly_forward_lanes<const L: usize>(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
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
        let t = crate::reduce::fqmul(zeta, hi[i]);
        hi[i] = lo[i] - t;
        lo[i] += t;
    }
}

/// Inverse butterfly: `(lo, hi) <- (barrett(lo+hi), zeta*(hi-lo)*R^{-1})`.
#[inline]
pub fn butterfly_inverse(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
    super::dispatch_lanes!(butterfly_inverse_lanes(lo, hi, zeta));
}

#[inline]
fn butterfly_inverse_lanes<const L: usize>(lo: &mut [i16], hi: &mut [i16], zeta: i16) {
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
        lo[i] = crate::reduce::barrett_reduce(t + hi[i]);
        hi[i] = crate::reduce::fqmul(zeta, hi[i] - t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_butterfly_forward_matches_scalar() {
        const M: usize = 32;
        let mut lo_s = [0i16; M];
        let mut hi_s = [0i16; M];
        for i in 0..M {
            lo_s[i] = (i as i16) * 13;
            hi_s[i] = (i as i16) * 7 + 100;
        }
        let mut lo_v = lo_s;
        let mut hi_v = hi_s;
        let zeta = 1234i16;

        butterfly_forward(&mut lo_v, &mut hi_v, zeta);
        for i in 0..M {
            let t = crate::reduce::fqmul(zeta, hi_s[i]);
            hi_s[i] = lo_s[i] - t;
            lo_s[i] += t;
        }
        assert_eq!(lo_v, lo_s);
        assert_eq!(hi_v, hi_s);
    }

    #[test]
    fn simd_butterfly_inverse_matches_scalar() {
        const M: usize = 32;
        let mut lo_s = [0i16; M];
        let mut hi_s = [0i16; M];
        for i in 0..M {
            lo_s[i] = (i as i16) * 11 - 200;
            hi_s[i] = (i as i16) * 5 + 50;
        }
        let mut lo_v = lo_s;
        let mut hi_v = hi_s;
        let zeta = -567i16;

        butterfly_inverse(&mut lo_v, &mut hi_v, zeta);
        for i in 0..M {
            let t = lo_s[i];
            lo_s[i] = crate::reduce::barrett_reduce(t + hi_s[i]);
            hi_s[i] = crate::reduce::fqmul(zeta, hi_s[i] - t);
        }
        assert_eq!(lo_v, lo_s);
        assert_eq!(hi_v, hi_s);
    }
}
