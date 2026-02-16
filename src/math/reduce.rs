//! Montgomery and Barrett modular reduction for the ML-KEM field (q = 3329).

use crate::params::Q;

/// q⁻¹ mod 2¹⁶ (Montgomery inverse).
pub const QINV: i16 = -3327;

/// 2¹⁶ mod q (Montgomery radix residue).
pub const MONT: i16 = -1044;

/// Montgomery reduction: computes `a · R⁻¹ mod q` where R = 2¹⁶.
///
/// Input:  `a ∈ {−q·2¹⁵, …, q·2¹⁵ − 1}`.
/// Output: `r ∈ {−q+1, …, q−1}` with `r ≡ a·R⁻¹ (mod q)`.
#[inline]
pub fn montgomery_reduce(a: i32) -> i16 {
    let t = (a as i16).wrapping_mul(QINV);
    ((a - (t as i32) * (Q as i32)) >> 16) as i16
}

/// Barrett reduction: centered reduction modulo q.
///
/// Input:  `a` with `|a| < 2q` (typical after butterfly addition).
/// Output: `r ∈ {−⌊q/2⌋, …, ⌊q/2⌋}` with `r ≡ a (mod q)`.
#[inline]
pub fn barrett_reduce(a: i16) -> i16 {
    const V: i32 = ((1i32 << 26) + (Q as i32) / 2) / (Q as i32); // 20159
    let t = ((V * (a as i32) + (1 << 25)) >> 26) as i16;
    a - t.wrapping_mul(Q)
}

/// Field multiplication followed by Montgomery reduction: `a·b·R⁻¹ mod q`.
#[inline]
pub fn fqmul(a: i16, b: i16) -> i16 {
    montgomery_reduce((a as i32) * (b as i32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn montgomery_reduce_of_zero() {
        assert_eq!(montgomery_reduce(0), 0);
    }

    #[test]
    fn barrett_reduce_small_positive() {
        // Value already in range should stay close
        let r = barrett_reduce(42);
        assert_eq!(r, 42);
    }

    #[test]
    fn barrett_reduce_wraps() {
        // Q should reduce to 0
        let r = barrett_reduce(Q);
        assert_eq!(r, 0);
    }

    #[test]
    fn fqmul_mont_identity() {
        // fqmul(a, MONT) = a * MONT * R^-1 = a * (R mod q) * R^-1 = a mod q
        // So fqmul(1, MONT) should give 1 (since 1 < q).
        let r = fqmul(1, MONT);
        // MONT * 1 * R^-1 mod q = R * R^-1 mod q = 1
        assert_eq!(r, 1);
    }

    #[test]
    fn barrett_reduce_negative() {
        let r = barrett_reduce(-Q);
        assert_eq!(r, 0);
    }
}
