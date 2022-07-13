use std::fmt::Debug;
use std::io;
use std::result;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::IO(err)
    }
}

pub type Result<T> = result::Result<T, Error>;
