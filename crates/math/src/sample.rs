//! Deterministic sampling: sealed CBD noise traits and rejection-uniform.

use crate::{N, Q};

mod sealed {
    pub trait Sealed {}
}

/// Sealed trait for CBD noise sampling width.
pub trait CbdWidth: sealed::Sealed {
    const ETA: usize;
    const BUF_BYTES: usize;

    fn sample(r: &mut [i16; N], buf: &[u8]);
}

pub struct Eta2;
pub struct Eta3;

impl sealed::Sealed for Eta2 {}
impl CbdWidth for Eta2 {
    const ETA: usize = 2;
    const BUF_BYTES: usize = 2 * N / 4;

    #[inline]
    fn sample(r: &mut [i16; N], buf: &[u8]) {
        debug_assert!(buf.len() >= 2 * N / 4);
        for i in 0..N / 8 {
            let t =
                u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
            let d = (t & 0x5555_5555) + ((t >> 1) & 0x5555_5555);
            for j in 0..8 {
                let a = ((d >> (4 * j)) & 3) as i16;
                let b = ((d >> (4 * j + 2)) & 3) as i16;
                r[8 * i + j] = a - b;
            }
        }
    }
}

impl sealed::Sealed for Eta3 {}
impl CbdWidth for Eta3 {
    const ETA: usize = 3;
    const BUF_BYTES: usize = 3 * N / 4;

    #[inline]
    fn sample(r: &mut [i16; N], buf: &[u8]) {
        debug_assert!(buf.len() >= 3 * N / 4);
        for i in 0..N / 4 {
            let t =
                u32::from_le_bytes([buf[3 * i], buf[3 * i + 1], buf[3 * i + 2], 0]) & 0x00FF_FFFF;
            let d = (t & 0x0024_9249) + ((t >> 1) & 0x0024_9249) + ((t >> 2) & 0x0024_9249);
            for j in 0..4 {
                let a = ((d >> (6 * j)) & 7) as i16;
                let b = ((d >> (6 * j + 3)) & 7) as i16;
                r[4 * i + j] = a - b;
            }
        }
    }
}

const SHAKE128_RATE: usize = 168;

pub fn reject_uniform(r: &mut [i16; N], mut fill: impl FnMut(&mut [u8])) -> usize {
    let mut ctr = 0;
    let mut buf = [0u8; SHAKE128_RATE];

    while ctr < N {
        fill(&mut buf);
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
        Eta2::sample(&mut r, &buf);
        for &c in &r {
            assert!(
                (-2..=2).contains(&c),
                "coefficient {c} out of range for eta=2"
            );
        }
    }

    #[test]
    fn cbd3_output_range() {
        let buf = [0x5Au8; 192];
        let mut r = [0i16; N];
        Eta3::sample(&mut r, &buf);
        for &c in &r {
            assert!(
                (-3..=3).contains(&c),
                "coefficient {c} out of range for eta=3"
            );
        }
    }

    #[test]
    fn cbd2_zero_input() {
        let buf = [0u8; 128];
        let mut r = [99i16; N];
        Eta2::sample(&mut r, &buf);
        assert!(r.iter().all(|&c| c == 0));
    }

    #[test]
    fn reject_uniform_fills_completely() {
        let mut counter = 0u8;
        let mut r = [0i16; N];
        let count = reject_uniform(&mut r, |buf| {
            for b in buf.iter_mut() {
                *b = counter;
                counter = counter.wrapping_add(1);
            }
        });
        assert_eq!(count, N);
        for &c in &r {
            assert!((0..Q).contains(&c), "coefficient {c} out of [0, q)");
        }
    }
}
