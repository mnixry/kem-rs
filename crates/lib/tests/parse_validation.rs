//! Negative tests for parse validation and the `Error` surface.

use kem_rs::{
    Ciphertext, Error, MlKem512, MlKem768, MlKem1024, ParameterSet, PublicKey, SecretKey,
    keypair_derand,
};

fn valid_keypair<P: ParameterSet>() -> (PublicKey<P>, SecretKey<P>) {
    let coins: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37));
    keypair_derand::<P>(&coins)
}

macro_rules! parse_tests {
    ($param:ty, $mod:ident) => {
        mod $mod {
            use super::*;

            #[test]
            fn pk_empty() {
                let r: Result<PublicKey<$param>, _> = (&[][..]).try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::PK_BYTES,
                        actual: 0,
                    }
                );
            }

            #[test]
            fn pk_short() {
                let buf = vec![0u8; <$param>::PK_BYTES - 1];
                let r: Result<PublicKey<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::PK_BYTES,
                        actual: <$param>::PK_BYTES - 1,
                    }
                );
            }

            #[test]
            fn pk_long() {
                let buf = vec![0u8; <$param>::PK_BYTES + 1];
                let r: Result<PublicKey<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::PK_BYTES,
                        actual: <$param>::PK_BYTES + 1,
                    }
                );
            }

            #[test]
            fn sk_empty() {
                let r: Result<SecretKey<$param>, _> = (&[][..]).try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::SK_BYTES,
                        actual: 0,
                    }
                );
            }

            #[test]
            fn sk_short() {
                let buf = vec![0u8; <$param>::SK_BYTES - 1];
                let r: Result<SecretKey<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::SK_BYTES,
                        actual: <$param>::SK_BYTES - 1,
                    }
                );
            }

            #[test]
            fn sk_long() {
                let buf = vec![0u8; <$param>::SK_BYTES + 1];
                let r: Result<SecretKey<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::SK_BYTES,
                        actual: <$param>::SK_BYTES + 1,
                    }
                );
            }

            #[test]
            fn ct_empty() {
                let r: Result<Ciphertext<$param>, _> = (&[][..]).try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::CT_BYTES,
                        actual: 0,
                    }
                );
            }

            #[test]
            fn ct_short() {
                let buf = vec![0u8; <$param>::CT_BYTES - 1];
                let r: Result<Ciphertext<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::CT_BYTES,
                        actual: <$param>::CT_BYTES - 1,
                    }
                );
            }

            #[test]
            fn ct_long() {
                let buf = vec![0u8; <$param>::CT_BYTES + 1];
                let r: Result<Ciphertext<$param>, _> = buf.as_slice().try_into();
                assert_eq!(
                    r.unwrap_err(),
                    Error::InvalidLength {
                        expected: <$param>::CT_BYTES,
                        actual: <$param>::CT_BYTES + 1,
                    }
                );
            }

            #[test]
            fn sk_corrupted_hash_is_invalid_key() {
                let (_, sk) = valid_keypair::<$param>();
                let mut buf = sk.as_ref().to_vec();
                // `h` is the second-to-last 32-byte field in the SK layout.
                let h_offset = <$param>::SK_BYTES - 64;
                buf[h_offset] ^= 0xFF;
                let r: Result<SecretKey<$param>, _> = buf.as_slice().try_into();
                assert_eq!(r.unwrap_err(), Error::InvalidKey);
            }

            #[test]
            fn valid_pk_parses() {
                let (pk, _) = valid_keypair::<$param>();
                let r: Result<PublicKey<$param>, _> = pk.as_ref().try_into();
                assert!(r.is_ok());
            }

            #[test]
            fn valid_sk_parses() {
                let (_, sk) = valid_keypair::<$param>();
                let r: Result<SecretKey<$param>, _> = sk.as_ref().try_into();
                assert!(r.is_ok());
            }
        }
    };
}

parse_tests!(MlKem512, mlkem512);
parse_tests!(MlKem768, mlkem768);
parse_tests!(MlKem1024, mlkem1024);

#[test]
fn error_display_invalid_length() {
    let err = Error::InvalidLength {
        expected: 800,
        actual: 42,
    };
    assert_eq!(err.to_string(), "invalid length: expected 800, got 42");
}

#[test]
fn error_display_invalid_key() {
    assert_eq!(Error::InvalidKey.to_string(), "invalid key");
}

#[test]
fn error_is_std_error() {
    use std::error::Error as _;
    assert!(kem_rs::Error::InvalidKey.source().is_none());
}
