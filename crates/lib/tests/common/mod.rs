use serde::{Deserialize, de};

pub fn parse_json<T: for<'de> Deserialize<'de>>(bytes: impl AsRef<[u8]>) -> T {
    serde_json::from_slice(bytes.as_ref()).expect("json parse failed")
}

fn de_hex_vec<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>, {
    let encoded = String::deserialize(deserializer)?;
    hex::decode(&encoded).map_err(de::Error::custom)
}

fn de_hex_array<'de, const N: usize, D>(deserializer: D) -> Result<[u8; N], D::Error>
where
    D: serde::Deserializer<'de>, {
    let bytes = de_hex_vec(deserializer)?;
    bytes
        .try_into()
        .map_err(|_| de::Error::custom("expected 32 bytes"))
}

#[derive(Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct HexBytes(#[serde(deserialize_with = "de_hex_vec")] Vec<u8>);

impl AsRef<[u8]> for HexBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::ops::Deref for HexBytes {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
pub struct HexArray<const N: usize>(#[serde(deserialize_with = "de_hex_array")] [u8; N]);

impl<const N: usize> AsRef<[u8; N]> for HexArray<N> {
    fn as_ref(&self) -> &[u8; N] {
        &self.0
    }
}

impl<const N: usize> std::ops::Deref for HexArray<N> {
    type Target = [u8; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
