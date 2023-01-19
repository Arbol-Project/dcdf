//! Extend Read and Write with some convenience methods for binary i/o
//!
use std::{io, marker::Unpin};

use async_trait::async_trait;
use futures::{io as aio, AsyncReadExt, AsyncWriteExt};

use super::errors::Result;

pub trait Serialize: Sized {
    /// Write self to a stream
    fn write_to(&self, stream: &mut impl io::Write) -> Result<()>;

    /// Read Self from a stream
    fn read_from(stream: &mut impl io::Read) -> Result<Self>;
}

#[async_trait]
pub trait SerializeAsync: Sized {
    /// Write self to a stream
    async fn write_to_async(
        &self,
        stream: &mut (impl aio::AsyncWrite + Unpin + Send),
    ) -> Result<()>;

    /// Read Self from a stream
    async fn read_from_async(stream: &mut (impl aio::AsyncRead + Unpin + Send)) -> Result<Self>;
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

#[async_trait]
pub trait ExtendedAsyncRead: aio::AsyncRead {
    /// Read a byte from a stream
    async fn read_byte_async(&mut self) -> io::Result<u8>;

    /// Read a Big Endian encoded 16 bit unsigned integer from a stream
    async fn read_u16_async(&mut self) -> io::Result<u16>;

    /// Read a Big Endian encoded 32 bit signed integer from a stream
    async fn read_i32_async(&mut self) -> io::Result<i32>;

    /// Read a Big Endian encoded 32 bit integer from a stream
    async fn read_u32_async(&mut self) -> io::Result<u32>;
}

#[async_trait]
impl<R: aio::AsyncRead + Unpin + Send> ExtendedAsyncRead for R {
    /// Read a byte from a stream
    async fn read_byte_async(&mut self) -> io::Result<u8> {
        let mut buffer = [0; 1];
        self.read_exact(&mut buffer).await?;

        Ok(buffer[0])
    }

    /// Read a Big Endian encoded 16 bit unsigned integer from a stream
    async fn read_u16_async(&mut self) -> io::Result<u16> {
        let mut buffer = [0; 2];
        self.read_exact(&mut buffer).await?;

        Ok(u16::from_be_bytes(buffer))
    }

    /// Read a Big Endian encoded 32 bit signed integer from a stream
    async fn read_i32_async(&mut self) -> io::Result<i32> {
        let mut buffer = [0; 4];
        self.read_exact(&mut buffer).await?;

        Ok(i32::from_be_bytes(buffer))
    }

    /// Read a Big Endian encoded 32 bit unsigned integer from a stream
    async fn read_u32_async(&mut self) -> io::Result<u32> {
        let mut buffer = [0; 4];
        self.read_exact(&mut buffer).await?;

        Ok(u32::from_be_bytes(buffer))
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

#[async_trait]
pub trait ExtendedAsyncWrite: aio::AsyncWrite {
    /// Write a byte to a stream
    async fn write_byte_async(&mut self, byte: u8) -> io::Result<()>;

    /// Write a Big Endian encoded 16 bit unsigned integer to a stream
    async fn write_u16_async(&mut self, word: u16) -> io::Result<()>;

    /// Write a Big Endian encoded 32 bit signed integer to a stream
    async fn write_i32_async(&mut self, word: i32) -> io::Result<()>;

    /// Write a Big Endian encoded 32 bit unsigned integer to a stream
    async fn write_u32_async(&mut self, word: u32) -> io::Result<()>;
}

#[async_trait]
impl<W: aio::AsyncWrite + Unpin + Send> ExtendedAsyncWrite for W {
    /// Write a byte to a stream
    async fn write_byte_async(&mut self, byte: u8) -> io::Result<()> {
        let buffer = [byte];
        self.write_all(&buffer).await?;

        Ok(())
    }

    /// Write a Big Endian encoded 16 bit unsigned integer to a stream
    async fn write_u16_async(&mut self, word: u16) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer).await?;

        Ok(())
    }

    /// Write a Big Endian encoded 32 bit signed integer to a stream
    async fn write_i32_async(&mut self, word: i32) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer).await?;

        Ok(())
    }

    /// Write a Big Endian encoded 32 bit unsigned integer to a stream
    async fn write_u32_async(&mut self, word: u32) -> io::Result<()> {
        let buffer = word.to_be_bytes();
        self.write_all(&buffer).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::io::Cursor as AsyncCursor;
    use std::io::Cursor;

    #[test]
    fn test_all_of_it() -> io::Result<()> {
        let mut buffer: Vec<u8> = Vec::new();
        buffer.write_byte(42)?;
        buffer.write_u16(41968)?;
        buffer.write_u32(31441968)?;
        buffer.write_i32(-31441968)?;

        let mut buffer = Cursor::new(buffer);
        assert_eq!(buffer.read_byte()?, 42);
        assert_eq!(buffer.read_u16()?, 41968);
        assert_eq!(buffer.read_u32()?, 31441968);
        assert_eq!(buffer.read_i32()?, -31441968);

        Ok(())
    }

    #[tokio::test]
    async fn test_all_of_it_async() -> io::Result<()> {
        let mut buffer: Vec<u8> = Vec::new();
        buffer.write_byte_async(42).await?;
        buffer.write_u16_async(41968).await?;
        buffer.write_u32_async(31441968).await?;
        buffer.write_i32_async(-31441968).await?;

        let mut buffer = AsyncCursor::new(buffer);
        assert_eq!(buffer.read_byte_async().await?, 42);
        assert_eq!(buffer.read_u16_async().await?, 41968);
        assert_eq!(buffer.read_u32_async().await?, 31441968);
        assert_eq!(buffer.read_i32_async().await?, -31441968);

        Ok(())
    }
}
