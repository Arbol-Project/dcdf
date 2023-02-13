use std::fmt;
use std::io;
use std::result;

use cid;
use unsigned_varint::io::ReadError as UnsignedVarintError;

#[derive(fmt::Debug)]
pub enum Error {
    Io(io::Error),
    Cid(cid::Error),
    UnsignedVarint(UnsignedVarintError),
    Load,
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<cid::Error> for Error {
    fn from(err: cid::Error) -> Self {
        Self::Cid(err)
    }
}

impl From<UnsignedVarintError> for Error {
    fn from(err: UnsignedVarintError) -> Self {
        Self::UnsignedVarint(err)
    }
}

pub type Result<T> = result::Result<T, Error>;
