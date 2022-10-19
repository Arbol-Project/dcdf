//! Extend Read and Write with some convenience methods for binary i/o
//!
use std::io;

use super::errors::Result;

pub trait Serialize: Sized {
    /// Write self to a stream
    fn write_to(&self, _stream: &mut impl io::Write) -> Result<()> {
        // SMELL
        unimplemented!("This object cannot be deserialized.");
    }

    /// Read Self from a stream
    fn read_from(_stream: &mut impl io::Read) -> Result<Self> {
        // SMELL
        unimplemented!("This object cannot be deserialized.");
    }
}

pub trait ExtendedRead: io::Read {
    /// Read a byte from a stream
    fn read_byte(&mut self) -> io::Result<u8>;

    /// Read a Big Endian encoded 16 bit unsigned integer from a stream
    fn read_u16(&mut self) -> io::Result<u16>;

    /// Read a Big Endian encoded 32 bit integer from a stream
    fn read_u32(&mut self) -> io::Result<u32>;

    /// Read a Big Endian encoded 32 bit signed integer from a stream
    fn read_i32(&mut self) -> io::Result<i32>;
}

impl<R: io::Read> ExtendedRead for R {
    /// Read a byte from a stream
    fn read_byte(&mut self) -> io::Result<u8> {
        let mut buffer = [0; 1];
        self.read_exact(&mut buffer)?;

        Ok(buffer[0])
    }

    /// Read a Big Endian encoded 16 bit unsigned integer from a stream
    fn read_u16(&mut self) -> io::Result<u16> {
        let mut buffer = [0; 2];
        self.read_exact(&mut buffer)?;

        Ok(u16::from_be_bytes(buffer))
    }

    /// Read a Big Endian encoded 32 bit unsigned integer from a stream
    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buffer = [0; 4];
        self.read_exact(&mut buffer)?;

        Ok(u32::from_be_bytes(buffer))
    }

    /// Read a Big Endian encoded 32 bit signed integer from a stream
    fn read_i32(&mut self) -> io::Result<i32> {
        let mut buffer = [0; 4];
        self.read_exact(&mut buffer)?;

        Ok(i32::from_be_bytes(buffer))
    }
}

pub trait ExtendedWrite: io::Write {
    /// Write a byte to a stream
    fn write_byte(&mut self, byte: u8) -> io::Result<()>;

    /// Write a Big Endian encoded 16 bit unsigned integer to a stream
    fn write_u16(&mut self, word: u16) -> io::Result<()>;

    /// Write a Big Endian encoded 32 bit unsigned integer to a stream
    fn write_u32(&mut self, word: u32) -> io::Result<()>;

    /// Write a Big Endian encoded 32 bit signed integer to a stream
    fn write_i32(&mut self, word: i32) -> io::Result<()>;
}

impl<W: io::Write> ExtendedWrite for W {
    /// Write a byte to a stream
    fn write_byte(&mut self, byte: u8) -> io::Result<()> {
        let buffer = [byte];
        self.write_all(&buffer)?;

        Ok(())
    }

    /// Write a Big Endian encoded 16 bit unsigned integer to a stream
    fn write_u16(&mut self, word: u16) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer)?;

        Ok(())
    }

    /// Write a Big Endian encoded 32 bit unsigned integer to a stream
    fn write_u32(&mut self, word: u32) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer)?;

        Ok(())
    }

    /// Write a Big Endian encoded 32 bit signed integer to a stream
    fn write_i32(&mut self, word: i32) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Seek;
    use tempfile::tempfile;

    #[test]
    fn test_all_of_it() -> io::Result<()> {
        let mut file = tempfile()?;
        file.write_byte(42)?;
        file.write_u16(41968)?;
        file.write_u32(31441968)?;
        file.write_i32(-31441968)?;

        file.sync_all()?;
        file.rewind()?;

        assert_eq!(file.read_byte()?, 42);
        assert_eq!(file.read_u16()?, 41968);
        assert_eq!(file.read_u32()?, 31441968);
        assert_eq!(file.read_i32()?, -31441968);

        Ok(())
    }
}
