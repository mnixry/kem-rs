//! Montgomery and Barrett modular reduction for the ML-KEM field (q = 3329).

use crate::Q;

/// q^{-1} mod 2^{16} (Montgomery inverse).
pub const QINV: i16 = -3327;

/// Montgomery reduction: computes `a * R^{-1} mod q` where R = 2^{16}.
///
/// Input: `a in {-q*2^{15}, ..., q*2^{15} - 1}`. Output: `r in {-q+1, ...,
/// q-1}` with `r \equiv a*R^{-1} (mod q)`.
#[inline]
#[must_use]
pub const fn montgomery_reduce(a: i32) -> i16 {
    let t = (a as i16).wrapping_mul(QINV);
    ((a - (t as i32) * (Q as i32)) >> 16) as i16
}

/// Barrett reduction: centered reduction modulo q.
///
/// Input `a` with `|a| < 2q` (typical after butterfly). Output: `r in
/// {-floor(q/2), ..., floor(q/2)}` with `r \equiv a (mod q)`.
#[inline]
#[must_use]
#[cfg(test)]
pub const fn barrett_reduce(a: i16) -> i16 {
    const V: i32 = ((1i32 << 26) + (Q as i32) / 2) / (Q as i32); // 20159
    let t = ((V * (a as i32) + (1 << 25)) >> 26) as i16;
    a - t.wrapping_mul(Q)
}

/// Field multiplication followed by Montgomery reduction: `a*b*R^{-1} mod q`.
#[inline]
#[must_use]
pub const fn fqmul(a: i16, b: i16) -> i16 {
    montgomery_reduce((a as i32) * (b as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2^{16} mod q (Montgomery radix residue).
    const MONT: i16 = -1044;

    #[test]
    fn montgomery_reduce_of_zero() {
        assert_eq!(montgomery_reduce(0), 0);
    }

    #[test]
    fn barrett_reduce_small_positive() {
        let r = barrett_reduce(42);
        assert_eq!(r, 42);
    }

    #[test]
    fn barrett_reduce_wraps() {
        let r = barrett_reduce(Q);
        assert_eq!(r, 0);
    }

    #[test]
    fn fqmul_mont_identity() {
        // fqmul(a, MONT) = a * MONT * R^-1 = a * (R mod q) * R^-1 = a mod q
        let r = fqmul(1, MONT);
        assert_eq!(r, 1);
    }

    #[test]
    fn barrett_reduce_negative() {
        let r = barrett_reduce(-Q);
        assert_eq!(r, 0);
    }
}
