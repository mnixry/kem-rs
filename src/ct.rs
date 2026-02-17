//! Constant-time comparison and conditional-move (verify/cmov). No secret-dependent branching.

/// Constant-time byte-slice comparison. Returns 0 if a == b, 1 otherwise. Same length required.
#[inline]
pub fn ct_verify(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "ct_verify: length mismatch");

    let mut diff: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= (x ^ y) as u64;
    }
    // Fence: prevent the optimiser from short-circuiting the loop.
    let diff = core::hint::black_box(diff);
    // Map 0 -> 0, nonzero -> 1 without branching.
    (diff.wrapping_neg() >> 63) as u8
}

/// Constant-time conditional copy. If condition==1 overwrites dst with src; if 0, dst unchanged. Panics if condition not 0/1 or lengths differ.
#[inline]
pub fn ct_cmov(dst: &mut [u8], src: &[u8], condition: u8) {
    assert_eq!(dst.len(), src.len(), "ct_cmov: length mismatch");
    debug_assert!(condition <= 1, "ct_cmov: condition must be 0 or 1");

    // Fence the condition so the mask isn't optimised into a branch.
    let mask = core::hint::black_box(condition).wrapping_neg(); // 0x00 or 0xFF

    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= mask & (*d ^ s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_equal() {
        let a = [0u8; 64];
        let b = [0u8; 64];
        assert_eq!(ct_verify(&a, &b), 0);
    }

    #[test]
    fn verify_differ_first_byte() {
        let a = [0u8; 64];
        let mut b = [0u8; 64];
        b[0] = 1;
        assert_eq!(ct_verify(&a, &b), 1);
    }

    #[test]
    fn verify_differ_last_byte() {
        let a = [0u8; 64];
        let mut b = [0u8; 64];
        b[63] = 0x80;
        assert_eq!(ct_verify(&a, &b), 1);
    }

    #[test]
    fn cmov_condition_zero_is_noop() {
        let mut dst = [0xAA_u8; 32];
        let src = [0xBB_u8; 32];
        ct_cmov(&mut dst, &src, 0);
        assert!(dst.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn cmov_condition_one_copies() {
        let mut dst = [0xAA_u8; 32];
        let src = [0xBB_u8; 32];
        ct_cmov(&mut dst, &src, 1);
        assert!(dst.iter().all(|&b| b == 0xBB));
    }

    #[test]
    fn verify_empty_slices() {
        assert_eq!(ct_verify(&[], &[]), 0);
    }
}
