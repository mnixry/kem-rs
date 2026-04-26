//! Deterministic sampling: sealed CBD noise traits and rejection-uniform.

use crate::{ByteArray, N};

mod sealed {
    pub trait Sealed {}
}

pub trait CbdWidthParams: sealed::Sealed {
    const ETA: usize;
    const BUF_BYTES: usize;
    type Buffer: ByteArray;
}

macro_rules! cbd_width {
    ($($name:ident: $eta:expr, $poly_bytes:expr),*) => {
        $(
            pub struct $name;
            impl sealed::Sealed for $name {}
            impl CbdWidthParams for $name {
                const ETA: usize = $eta;
                const BUF_BYTES: usize = $poly_bytes;
                type Buffer = [u8; $poly_bytes];
            }
        )*
    };
}

cbd_width!(
    Eta2: 2, 2 * N / 4,
    Eta3: 3, 3 * N / 4
);

/// Sealed trait for CBD noise sampling width.
pub trait CbdWidth: CbdWidthParams {
    fn sample(r: &mut [i16; N], buf: &<Self as CbdWidthParams>::Buffer);
}

impl CbdWidth for Eta2 {
    #[inline(always)]
    fn sample(r: &mut [i16; N], buf: &<Self as CbdWidthParams>::Buffer) {
        debug_assert!(buf.len() >= 2 * N / 4);
        let (r_chunks, _) = r.as_chunks_mut();
        let (buf_chunks, _) = buf.as_chunks();
        for (chunk, &buf_chunk) in r_chunks.iter_mut().zip(buf_chunks) {
            let t = u32::from_le_bytes(buf_chunk);
            let d = (t & 0x5555_5555) + ((t >> 1) & 0x5555_5555);
            *chunk = unroll!(j in [..8], {
                let a = ((d >> (4 * j)) & 3) as i16;
                let b = ((d >> (4 * j + 2)) & 3) as i16;
                a - b
            });
        }
    }
}

impl CbdWidth for Eta3 {
    #[inline(always)]
    fn sample(r: &mut [i16; N], buf: &<Self as CbdWidthParams>::Buffer) {
        debug_assert!(buf.len() >= 3 * N / 4);
        let (r_chunks, _) = r.as_chunks_mut();
        let (buf_chunks, _) = buf.as_chunks::<3>();
        for (chunk, &[b0, b1, b2]) in r_chunks.iter_mut().zip(buf_chunks) {
            let t = u32::from_le_bytes([b0, b1, b2, 0]) & 0x00FF_FFFF;
            let d = (t & 0x0024_9249) + ((t >> 1) & 0x0024_9249) + ((t >> 2) & 0x0024_9249);
            *chunk = unroll!(j in [..4], {
                let a = ((d >> (6 * j)) & 7) as i16;
                let b = ((d >> (6 * j + 3)) & 7) as i16;
                a - b
            });
        }
    }
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
}
