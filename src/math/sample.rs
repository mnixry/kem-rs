//! Deterministic sampling: CBD noise ([`cbd2`], [`cbd3`]) and rejection-uniform ([`rej_uniform`]).

use crate::params::{N, Q};
use sha3::digest::XofReader;

/// SHAKE-128 output rate in bytes (one Keccak-f[1600] squeeze).
pub const SHAKE128_RATE: usize = 168;

/// CBD with eta=2: 128 bytes of PRF output -> 256 coefficients in {-2, ..., 2}.
pub fn cbd2(r: &mut [i16; N], buf: &[u8]) {
    debug_assert!(buf.len() >= 2 * N / 4); // 128 bytes
    for i in 0..N / 8 {
        let t = u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
        let d = (t & 0x5555_5555) + ((t >> 1) & 0x5555_5555);
        for j in 0..8 {
            let a = ((d >> (4 * j)) & 3) as i16;
            let b = ((d >> (4 * j + 2)) & 3) as i16;
            r[8 * i + j] = a - b;
        }
    }
}

/// CBD with eta=3: 192 bytes of PRF output -> 256 coefficients in {-3, ..., 3}.
pub fn cbd3(r: &mut [i16; N], buf: &[u8]) {
    debug_assert!(buf.len() >= 3 * N / 4); // 192 bytes
    for i in 0..N / 4 {
        let t = u32::from_le_bytes([buf[3 * i], buf[3 * i + 1], buf[3 * i + 2], 0]) & 0x00FF_FFFF;
        let d = (t & 0x0024_9249) + ((t >> 1) & 0x0024_9249) + ((t >> 2) & 0x0024_9249);
        for j in 0..4 {
            let a = ((d >> (6 * j)) & 7) as i16;
            let b = ((d >> (6 * j + 3)) & 7) as i16;
            r[4 * i + j] = a - b;
        }
    }
}

/// Rejection-sample N uniformly random coefficients in [0, q) from SHAKE-128 XOF. Returns N.
pub fn rej_uniform(r: &mut [i16; N], xof: &mut impl XofReader) -> usize {
    let mut ctr = 0;
    let mut buf = [0u8; SHAKE128_RATE];

    while ctr < N {
        xof.read(&mut buf);
        let mut pos = 0;
        while ctr < N && pos + 3 <= SHAKE128_RATE {
            let val0 = ((buf[pos] as u16) | ((buf[pos + 1] as u16) << 8)) & 0x0FFF;
            let val1 = ((buf[pos + 1] as u16) >> 4) | ((buf[pos + 2] as u16) << 4);
            pos += 3;
            if val0 < Q as u16 {
                r[ctr] = val0 as i16;
                ctr += 1;
            }
            if ctr < N && val1 < Q as u16 {
                r[ctr] = val1 as i16;
                ctr += 1;
            }
        }
    }
    ctr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbd2_output_range() {
        let buf = [0xA5u8; 128];
        let mut r = [0i16; N];
        cbd2(&mut r, &buf);
        for &c in &r {
            assert!((-2..=2).contains(&c), "coefficient {c} out of range for eta=2");
        }
    }

    #[test]
    fn cbd3_output_range() {
        let buf = [0x5Au8; 192];
        let mut r = [0i16; N];
        cbd3(&mut r, &buf);
        for &c in &r {
            assert!((-3..=3).contains(&c), "coefficient {c} out of range for eta=3");
        }
    }

    #[test]
    fn cbd2_zero_input() {
        let buf = [0u8; 128];
        let mut r = [99i16; N];
        cbd2(&mut r, &buf);
        // All-zero PRF output -> a = b = 0 -> all coefficients zero.
        assert!(r.iter().all(|&c| c == 0));
    }

    #[test]
    fn rej_uniform_fills_completely() {
        use crate::hash::xof_absorb;
        let seed = [42u8; 32];
        let mut xof = xof_absorb(&seed, 0, 0);
        let mut r = [0i16; N];
        let count = rej_uniform(&mut r, &mut xof);
        assert_eq!(count, N);
        // All coefficients in [0, q)
        for &c in &r {
            assert!((0..Q).contains(&c), "coefficient {c} out of [0, q)");
        }
    }
}
