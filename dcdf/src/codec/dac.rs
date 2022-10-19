//! Directly Accessible Code(s)
//!
//! A data structure which compresses integers using a variable length encoding that depends on the
//! integer's magnitude. See Ladra[^bib1], section 2.2.2 for more information.
//!
//! Signed integers are converted from twos complement to "zig-zag encoding"[^two] so that negative
//! numbers can be stored in as few bytes as possible.
//!
//! [^bib1]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
//!     structure for raster data, Information Systems 72 (2017) 179-204.
//!
//! [^two]: [ZigZag encoding/decoding explained](
//!     https://gist.github.com/mfuerstenau/ba870a29e16536fdbaba)
//!
use num_traits::PrimInt;
use std::fmt::Debug;
use std::io::{Read, Write};

use crate::cache::Cacheable;
use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite, Serialize};

use super::bitmap::{BitMap, BitMapBuilder};

/// Compact storage for integers (Directly Addressable Codes)
pub struct Dac {
    pub levels: Vec<(BitMap, Vec<u8>)>,
}

impl Serialize for Dac {
    /// Write the dac to a stream
    ///
    fn write_to(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_byte(self.levels.len() as u8)?;
        for (bitmap, bytes) in &self.levels {
            bitmap.write_to(stream)?;
            stream.write_all(bytes)?;
        }
        Ok(())
    }

    /// Read the dac from a stream
    ///
    fn read_from(stream: &mut impl Read) -> Result<Self> {
        let n_levels = stream.read_byte()? as usize;
        let mut levels = Vec::with_capacity(n_levels);
        for _ in 0..n_levels {
            let bitmap = BitMap::read_from(stream)?;
            let mut bytes = Vec::with_capacity(bitmap.length);
            stream.take(bitmap.length as u64).read_to_end(&mut bytes)?;

            levels.push((bitmap, bytes));
        }

        Ok(Self { levels })
    }
}

impl Cacheable for Dac {
    /// Get number of bytes of serialized dac
    fn size(&self) -> u64 {
        1 + self
            .levels
            .iter()
            .map(|(bitmap, bytes)| bitmap.size() + bytes.len() as u64)
            .sum::<u64>()
    }
}

impl Dac {
    /// Get the value at the given index
    ///
    pub fn get<I>(&self, index: usize) -> I
    where
        I: PrimInt + Debug,
    {
        let mut index = index;
        let mut n: u64 = 0;
        for (i, (bitmap, bytes)) in self.levels.iter().enumerate() {
            n |= (bytes[index] as u64) << i * 8;
            if bitmap.get(index) {
                index = bitmap.rank(index);
            } else {
                break;
            }
        }

        let n = zigzag_decode(n);
        I::from(n).unwrap()
    }
}

impl<I> From<Vec<I>> for Dac
where
    I: PrimInt + Debug,
{
    /// Compress a vector of integers into a dac data structure
    fn from(data: Vec<I>) -> Self {
        // Set up levels. Probably won't need all of them
        let mut levels = Vec::with_capacity(8);
        for _ in 0..8 {
            levels.push((BitMapBuilder::new(), Vec::new()));
        }

        // Chunk each datum into bytes, one per level, stopping when only 0s are left
        for datum in data {
            let mut datum = zigzag_encode(datum.to_i64().unwrap());
            for (bitmap, bytes) in &mut levels {
                bytes.push((datum & 0xff) as u8);
                datum >>= 8;
                if datum == 0 {
                    bitmap.push(false);
                    break;
                } else {
                    bitmap.push(true);
                }
            }
        }

        // Index bitmaps and prepare to return, stopping as soon as an empty level is encountered
        let levels = levels
            .into_iter()
            .take_while(|(bitmap, _)| bitmap.length > 0)
            .map(|(bitmap, bytes)| (bitmap.finish(), bytes))
            .collect();

        Dac { levels }
    }
}

fn zigzag_encode(n: i64) -> u64 {
    let zz = (n >> 63) ^ (n << 1);
    zz as u64
}

fn zigzag_decode(zz: u64) -> i64 {
    let n = (zz >> 1) ^ if zz & 1 == 1 { 0xffffffffffffffff } else { 0 };
    n as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Dac {
        // Some functions that are useful for testing but not needed otherwise.

        /// Get number of integers stored in this dac
        fn len(&self) -> usize {
            self.levels[0].0.length
        }

        /// Decompress back into Vector of integers
        pub fn collect<I>(&self) -> Vec<I>
        where
            I: PrimInt + Debug,
        {
            (0..self.len()).into_iter().map(|i| self.get(i)).collect()
        }
    }

    #[test]
    fn get_i32() {
        let data = vec![0, 2, -3, -2.pow(9), 2.pow(17) + 1, -2.pow(30) - 42];
        let dac = Dac::from(data.clone());
        for i in 0..data.len() {
            assert_eq!(dac.get::<i32>(i), data[i]);
        }
        assert_eq!(dac.levels[0].0.get(2), false);
    }

    #[test]
    fn this_one() {
        let data: Vec<i32> = vec![-512];
        let dac = Dac::from(data.clone());
        assert_eq!(zigzag_decode(zigzag_encode(-512)), -512);
        assert_eq!(dac.get::<i32>(0), data[0]);
    }
}
