use num_traits::PrimInt;
use std::fmt::Debug;
use std::io::{Read, Write};

use crate::cache::Cacheable;
use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite, Serialize};

/// Used to build up a BitMap.
///
/// Bits are pushed onto a bitmap one at a time using the `push` method. Use the `finish` method to
/// convert to a BitMap.
///
/// # Example
///
/// let mut builder = BitMapBuilder::new();
///
/// builder.push(true);
/// builder.push(false);
/// builder.push(true);
/// builder.push(false);
///
/// let bitmap = builder.finish();
///
pub struct BitMapBuilder {
    pub length: usize,
    bitmap: Vec<u8>,
}

impl BitMapBuilder {
    /// Initialize an empty BitMapBuilder
    pub fn new() -> BitMapBuilder {
        BitMapBuilder {
            length: 0,
            bitmap: vec![],
        }
    }

    /// Push a bit onto the BitMapBuilder
    pub fn push(&mut self, bit: bool) {
        // Which bit do we need to set in the relevant byte?
        let position = self.length % 8;

        // How much do we need to shift to the left to get to that position?
        let shift = 7 - position;

        // If position == 0, we start a new byte
        if position == 0 {
            self.bitmap.push(if bit { 1 << shift } else { 0 });
        }
        // Otherwise add bit to currently started byte
        else if bit {
            let last = self.bitmap.len() - 1;
            self.bitmap[last] += 1 << shift;
        }

        self.length += 1;
    }

    /// Finish building BitMap.
    ///
    pub fn finish(self) -> BitMap {
        // Value of k is more or less arbitrary. Could be tuned via benchmarking.
        // Index will add bitmap.length / k extra space to store the index
        let k = 4; // 25% extra space to store the index
        let blocks = self.length / 32 / k;
        let mut index: Vec<u32> = Vec::with_capacity(blocks);

        // Convert vector of u8 to vector of u32
        let words = div_ceil(self.length, 32);
        let mut bitmap32: Vec<u32> = Vec::with_capacity(words);
        if words > 0 {
            let mut shift = 24;
            let mut word_index = 0;

            for byte in self.bitmap {
                if shift == 24 {
                    bitmap32.push(0);
                }
                let mut word: u32 = byte.into();
                word <<= shift;
                bitmap32[word_index] |= word;

                if shift == 0 {
                    word_index += 1;
                    shift = 24;
                } else {
                    shift -= 8;
                }
            }
        }

        // Generate index
        let mut count = 0;
        for i in 0..blocks {
            for j in 0..k {
                count += bitmap32[i * k + j].count_ones();
            }
            index.push(count);
        }

        BitMap {
            length: self.length,
            k,
            index,
            bitmap: bitmap32,
        }
    }
}

/// An array of bits with a single level index for making fast rank queries.
///
pub struct BitMap {
    pub length: usize,
    k: usize,
    index: Vec<u32>,
    pub bitmap: Vec<u32>,
}

impl Serialize for BitMap {
    /// Write the bitmap to a stream
    ///
    fn write_to(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_u32(self.length as u32)?;
        stream.write_u32(self.k as u32)?;
        for index_block in &self.index {
            stream.write_u32(*index_block)?;
        }
        for bitmap_block in &self.bitmap {
            stream.write_u32(*bitmap_block)?;
        }
        Ok(())
    }

    /// Read a bitmap from a stream
    ///
    fn read_from(stream: &mut impl Read) -> Result<Self> {
        let length = stream.read_u32()? as usize;
        let k = stream.read_u32()? as usize;

        let blocks = length / 32 / k;
        let mut index = Vec::with_capacity(blocks as usize);
        for _ in 0..blocks {
            index.push(stream.read_u32()?);
        }

        let words = div_ceil(length, 32);
        let mut bitmap = Vec::with_capacity(words);
        for _ in 0..words {
            bitmap.push(stream.read_u32()?);
        }

        Ok(Self {
            length,
            k,
            index,
            bitmap,
        })
    }
}

impl Cacheable for BitMap {
    /// Return number of bytes in serialized representation
    ///
    fn size(&self) -> u64 {
        (4 + 4 + self.index.len() * 4 + self.bitmap.len() * 4) as u64
    }
}

impl BitMap {
    /// Get the bit at position `i`
    pub fn get(&self, i: usize) -> bool {
        let word_index = i / 32;
        let bit_index = i % 32;
        let shift = 31 - bit_index;
        let word = self.bitmap[word_index];

        (word >> shift) & 1 > 0
    }

    /// Count occurences of 1 in BitMap[0...i]
    pub fn rank(&self, i: usize) -> usize {
        if i > self.length {
            // Can only happen if there is a programming error in this module
            panic!("index out of bounds. length: {}, i: {}", self.length, i);
        }

        // Use the index
        let block = i / 32 / self.k;
        let mut count = if block > 0 { self.index[block - 1] } else { 0 };

        // Use popcount/count_ones on any whole words not included in index
        let start = block * self.k;
        let end = i / 32;
        for word in &self.bitmap[start..end] {
            count += word.count_ones();
        }

        // Count last bits in remaining fraction of a word
        let leftover_bits = i - end * 32;
        if leftover_bits > 0 {
            let word = &self.bitmap[end];
            let shift = 32 - leftover_bits;
            count += (word >> shift).count_ones();
        }

        count.try_into().unwrap()
    }

    /// Count occurences of 0 in BitMap[0...i]
    pub fn rank0(&self, i: usize) -> usize {
        i - self.rank(i)
    }
}

/// Returns n / m with remainder rounded up to nearest integer
fn div_ceil<I>(m: I, n: I) -> I
where
    I: PrimInt + Debug,
{
    let a = m / n;
    if m % n > I::zero() {
        a + I::one()
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::{Rng, RngCore};
    use std::time;

    impl BitMapBuilder {
        /// Count occurences of 1 in BitMap[0...i]
        ///
        /// Naive brute force implementation that is used to double check the indexed
        /// implementation in tests.
        pub fn rank(&self, i: usize) -> usize {
            let mut count = 0;
            for word in self.bitmap.iter().take(i / 8) {
                count += word.count_ones();
            }
            let leftover_bits = i % 8;
            if leftover_bits > 0 {
                let shift = 8 - leftover_bits;
                let word = self.bitmap[i / 8];
                count += (word >> shift).count_ones();
            }
            count.try_into().unwrap()
        }
    }

    #[test]
    fn from_bitmap() {
        let builder = BitMapBuilder {
            length: 36,
            bitmap: vec![99, 104, 114, 105, 115],
        };

        let bitmap = builder.finish();
        assert_eq!(bitmap.bitmap, vec![1667789417, 1929379840]);

        let builder = BitMapBuilder {
            length: 129,
            bitmap: vec![
                99, 104, 114, 105, 115, 0, 0, 0, 99, 104, 114, 105, 115, 0, 0, 0, 128,
            ],
        };

        let bitmap = builder.finish();
        assert_eq!(
            bitmap.bitmap,
            vec![1667789417, 1929379840, 1667789417, 1929379840, 1 << 31]
        );
        assert_eq!(bitmap.index, vec![40]);
    }

    fn test_rank(builder: BitMapBuilder, indexes: &[usize]) {
        // Gather answers using the naive, reference implementation
        println!(
            "Test rank: {} bits, {}/{} lookups",
            builder.length,
            indexes.len(),
            indexes.len() * 1000,
        );

        let timer = time::Instant::now();
        let mut answers: Vec<usize> = Vec::with_capacity(indexes.len());
        for index in indexes {
            answers.push(builder.rank(*index));
        }
        let reference_impl = timer.elapsed().as_millis();

        // Compare our answers with the reference implementation
        let timer = time::Instant::now();
        let bitmap = builder.finish();
        let make_index = timer.elapsed().as_millis();

        let timer = time::Instant::now();
        for _ in 1..1000 {
            for (index, answer) in indexes.iter().zip(answers.iter()) {
                assert_eq!(*answer, bitmap.rank(*index));
            }
        }
        let our_impl = timer.elapsed().as_millis();

        println!("time to build index: {}", make_index);
        println!("reference impl: {}, our impl: {}", reference_impl, our_impl);
    }

    #[test]
    fn get() {
        let answers = [
            true, false, true, false, true, false, true, false, false, false, true,
        ];
        let mut builder = BitMapBuilder::new();

        for answer in answers {
            builder.push(answer);
        }

        let bitmap = builder.finish();
        for (index, answer) in answers.iter().enumerate() {
            assert_eq!(bitmap.get(index), *answer);
        }
    }

    #[test]
    fn rank() {
        // 1010101001
        let bitmap = BitMapBuilder {
            length: 10,
            bitmap: vec![170, 64],
        };

        let indexes: Vec<usize> = (0..10).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[should_panic]
    fn rank_out_of_bounds() {
        // 1010101001
        let builder = BitMapBuilder {
            length: 10,
            bitmap: vec![170, 64],
        };

        let bitmap = builder.finish();
        bitmap.rank(11);
    }

    struct RandomRange(usize);

    impl Iterator for RandomRange {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            Some(rand::thread_rng().gen_range(0..self.0))
        }
    }

    fn make_bitmap(n: usize) -> BitMapBuilder {
        let timer = time::Instant::now();
        let mut bytes: Vec<u8> = Vec::with_capacity(n >> 3);
        bytes.resize(n >> 3, 0);
        rand::thread_rng().fill_bytes(&mut bytes);

        let bitmap = BitMapBuilder {
            length: n,
            bitmap: bytes,
        };
        println!("Time to build bitmap: {}", timer.elapsed().as_millis());

        bitmap
    }

    #[test]
    fn rank_megabit() {
        let bitmap = make_bitmap(1 << 20);
        let indexes = RandomRange(1 << 20);
        let indexes: Vec<usize> = indexes.take(100).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[ignore]
    fn rank_gigabit() {
        let bitmap = make_bitmap(1 << 30);
        let indexes: Vec<usize> = RandomRange(1 << 30).take(1000).collect();
        test_rank(bitmap, &indexes);
    }
}
