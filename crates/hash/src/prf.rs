//! SHAKE-256 PRF, generic over parallel lane count.

use core::simd::Simd;

use kem_math::{ByteArray, CbdWidth, SYMBYTES};

use crate::{
    SHAKE_PAD, SHAKE256_RATE,
    keccak::{PLEN, absorb_seed, f1600},
};

/// `K`-way parallel SHAKE-256 PRF: absorb `seed || nonce` per lane and
/// squeeze `Eta::BUF_BYTES` bytes from each.
#[must_use]
#[inline]
pub fn prf_batch<Eta: CbdWidth, const K: usize>(
    seed: &[u8; SYMBYTES], nonces: [u8; K],
) -> [Eta::Buffer; K] {
    const { assert!(Eta::BUF_BYTES % 8 == 0) };

    let mut outputs: [_; K] = core::array::from_fn(|_| Eta::Buffer::zeroed());

    let mut state: [Simd<u64, K>; PLEN] = [Simd::splat(0); PLEN];
    absorb_seed(&mut state, seed);

    state[4] = Simd::from_array(core::array::from_fn(|lane| {
        u64::from(nonces[lane]) | (u64::from(SHAKE_PAD) << 8)
    }));

    // End-of-rate padding: byte 135 = word 16, byte offset 7.
    state[16] = Simd::splat(0x80_u64 << 56);

    f1600(&mut state);

    let mut written = 0;
    while written < Eta::BUF_BYTES {
        let words = (Eta::BUF_BYTES - written).min(SHAKE256_RATE) / 8;

        for (i, s) in state.iter().enumerate().take(words) {
            let off = written + i * 8;
            let lanes = s.to_array();
            for (j, val) in lanes.iter().enumerate() {
                outputs[j].as_mut()[off..off + 8].copy_from_slice(&val.to_le_bytes());
            }
        }

        written += words * 8;
        if written < Eta::BUF_BYTES {
            f1600(&mut state);
        }
    }

    outputs
}
