//! Verify the parallel Keccak paths produce lane outputs identical
//! to the scalar SHAKE implementations, across all lane widths.

use kem_hash::{SHAKE128_RATE, prf_batch, shake128, shake256, xof_absorb};
use kem_math::{CbdWidthParams, Eta2, Eta3, SYMBYTES};

fn scalar_shake128_squeeze(seed: &[u8; SYMBYTES], x: u8, y: u8, n_blocks: usize) -> Vec<u8> {
    let mut input = [0u8; SYMBYTES + 2];
    input[..SYMBYTES].copy_from_slice(seed);
    input[SYMBYTES] = x;
    input[SYMBYTES + 1] = y;
    let mut out = vec![0u8; n_blocks * SHAKE128_RATE];
    shake128(input, &mut out);
    out
}

fn scalar_prf_shake256(seed: &[u8; SYMBYTES], nonce: u8, out_len: usize) -> Vec<u8> {
    let mut input = [0u8; SYMBYTES + 1];
    input[..SYMBYTES].copy_from_slice(seed);
    input[SYMBYTES] = nonce;
    let mut out = vec![0u8; out_len];
    shake256(input, &mut out);
    out
}

#[test]
fn xof_4_lanes_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x37));
    let indices = [(0, 0), (0, 1), (1, 0), (1, 1)];

    let mut reader = xof_absorb(&seed, indices);

    let n_squeezes = 4;
    let total_bytes = n_squeezes * SHAKE128_RATE;
    let mut outputs: [Vec<u8>; 4] = core::array::from_fn(|_| Vec::with_capacity(total_bytes));

    for _ in 0..n_squeezes {
        let blocks = reader.squeeze_blocks();
        for (lane, block) in outputs.iter_mut().zip(blocks.iter()) {
            lane.extend_from_slice(block);
        }
    }

    for (lane_idx, &(x, y)) in indices.iter().enumerate() {
        let expected = scalar_shake128_squeeze(&seed, x, y, n_squeezes);
        assert_eq!(
            outputs[lane_idx], expected,
            "xof<4> lane {lane_idx} (x={x}, y={y}) diverges from scalar SHAKE-128"
        );
    }
}

#[test]
fn xof_2_lanes_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x37));
    let indices = [(0, 0), (1, 1)];

    let mut reader = xof_absorb(&seed, indices);
    let blocks = reader.squeeze_blocks();

    for (lane_idx, &(x, y)) in indices.iter().enumerate() {
        let expected = scalar_shake128_squeeze(&seed, x, y, 1);
        assert_eq!(
            &blocks[lane_idx][..],
            &expected[..SHAKE128_RATE],
            "xof<2> lane {lane_idx} mismatch"
        );
    }
}

#[test]
fn xof_various_seeds() {
    for tag in 0..16u8 {
        let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_add(tag * 17));
        let indices = [(tag, 0), (0, tag), (tag, tag), (0, 0)];

        let mut reader = xof_absorb(&seed, indices);
        let blocks = reader.squeeze_blocks();

        for (lane_idx, &(x, y)) in indices.iter().enumerate() {
            let expected = scalar_shake128_squeeze(&seed, x, y, 1);
            assert_eq!(
                &blocks[lane_idx][..],
                &expected[..SHAKE128_RATE],
                "seed tag={tag} lane {lane_idx} mismatch"
            );
        }
    }
}

#[test]
fn prf_4_lanes_eta2_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x53));
    let nonces = [0u8, 1, 2, 3];
    let results = prf_batch::<Eta2, 4>(&seed, nonces);

    for (lane, &nonce) in nonces.iter().enumerate() {
        let expected = scalar_prf_shake256(&seed, nonce, Eta2::BUF_BYTES);
        assert_eq!(
            results[lane].as_ref(),
            &expected[..],
            "prf_batch<Eta2, 4> lane {lane} (nonce={nonce}) mismatch"
        );
    }
}

#[test]
fn prf_4_lanes_eta3_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x71));
    let nonces = [10u8, 20, 30, 40];
    let results = prf_batch::<Eta3, 4>(&seed, nonces);

    for (lane, &nonce) in nonces.iter().enumerate() {
        let expected = scalar_prf_shake256(&seed, nonce, Eta3::BUF_BYTES);
        assert_eq!(
            results[lane].as_ref(),
            &expected[..],
            "prf_batch<Eta3, 4> lane {lane} (nonce={nonce}) mismatch"
        );
    }
}

#[test]
fn prf_2_lanes_eta2_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x53));
    let nonces = [0u8, 1];
    let results = prf_batch::<Eta2, 2>(&seed, nonces);

    for (lane, &nonce) in nonces.iter().enumerate() {
        let expected = scalar_prf_shake256(&seed, nonce, Eta2::BUF_BYTES);
        assert_eq!(
            results[lane].as_ref(),
            &expected[..],
            "prf_batch<Eta2, 2> lane {lane} (nonce={nonce}) mismatch"
        );
    }
}

#[test]
fn prf_3_lanes_eta2_matches_scalar() {
    let seed: [u8; SYMBYTES] = core::array::from_fn(|i| (i as u8).wrapping_mul(0x53));
    let nonces = [0u8, 1, 2];
    let results = prf_batch::<Eta2, 3>(&seed, nonces);

    for (lane, &nonce) in nonces.iter().enumerate() {
        let expected = scalar_prf_shake256(&seed, nonce, Eta2::BUF_BYTES);
        assert_eq!(
            results[lane].as_ref(),
            &expected[..],
            "prf_batch<Eta2, 3> lane {lane} (nonce={nonce}) mismatch"
        );
    }
}

#[test]
fn prf_various_seeds_4_lanes() {
    for tag in 0..16u8 {
        let seed: [u8; SYMBYTES] =
            core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(41)));
        let nonces = [
            tag,
            tag.wrapping_add(1),
            tag.wrapping_add(2),
            tag.wrapping_add(3),
        ];

        let results = prf_batch::<Eta2, 4>(&seed, nonces);
        for (lane, &nonce) in nonces.iter().enumerate() {
            let expected = scalar_prf_shake256(&seed, nonce, Eta2::BUF_BYTES);
            assert_eq!(
                results[lane].as_ref(),
                &expected[..],
                "prf_batch tag={tag} lane {lane} mismatch"
            );
        }
    }
}
