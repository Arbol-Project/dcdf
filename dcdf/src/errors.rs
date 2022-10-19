use std::fmt::Debug;
use std::io;
use std::result;

use cid;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    CID(cid::Error),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::IO(err)
    }
}

impl From<cid::Error> for Error {
    fn from(err: cid::Error) -> Self {
        Self::CID(err)
    }
}

pub type Result<T> = result::Result<T, Error>;
