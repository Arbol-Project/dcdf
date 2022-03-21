//! Encode/Decode Heuristic K^2 Raster
//!
//! An implementation of the algorithm proposed by Silva-Coira, Paramá, de Bernardo, and Seco in
//! their paper, "Space-efficient representations of raster time series".
//!
//! See: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf

use num::{one, zero, Integer};

/// An array of bits.
///
/// This unindexed version is used to build up a BitMap using the `push` method. Once a BitMap is
/// built, it should be converted to an indexed type for performant rank and select queries.
///
/// Typical usage:
///
/// let builder = BitMap::new();
///
/// builder.push(...)
/// builder.push(...)
/// etc...
///
/// let mut bitmap = IndexedBitMap::from(builder);
///
pub struct BitMap {
    length: usize,
    bitmap: Vec<u8>,
}

impl BitMap {
    /// Initialize an empty BitMap
    pub fn new() -> BitMap {
        BitMap {
            length: 0,
            bitmap: vec![],
        }
    }

    /// Push a bit onto the BitMap
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

    /// Count occurences of 1 in BitMap[0...i]
    ///
    /// Naive brute force implementation that is only used to double check the indexed
    /// implementation in tests.
    pub fn rank(&self, i: usize) -> usize {
        if i > self.length {
            // Can only happen if there is a programming error in this module
            panic!("index out of bounds. length: {}, i: {}", self.length, i);
        }

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

    /// Get the index of the nth occurence of 1 in BitMap
    ///
    /// Naive brute force implementation that is only used to double check the indexed
    /// implementation in tests.
    pub fn select(&self, n: usize) -> Option<usize> {
        if n == 0 {
            panic!("select(0)");
        }

        let mut count = 0;
        for (word_index, word) in self.bitmap.iter().enumerate() {
            let popcount: usize = word.count_ones().try_into().unwrap();
            if popcount + count >= n {
                // It's in this word somewhere
                let mut position = word_index * 8;
                let mut mask = 1 << 7;
                while count < n {
                    if word & mask > 0 {
                        count += 1;
                    }
                    mask >>= 1;
                    position += 1;
                }
                return Some(position);
            }
            count += popcount;
        }
        None
    }
}

/// An array of bits with a single level index for making fast rank and select queries.
///
pub struct IndexedBitMap {
    length: usize,
    k: usize,
    index: Vec<u32>,
    bitmap: Vec<u32>,
}

impl From<BitMap> for IndexedBitMap {
    /// Generate an indexed bitmap from an unindexed one.
    ///
    /// Index is an array of bit counts for every k words in the bitmap, such that
    /// rank(i) = index[i / k / wordlen] if i is an even multiple of k * wordlen. wordlen is 32 for
    /// this implementation which uses 32 bit unsigned integers.
    fn from(bitmap: BitMap) -> Self {
        // Value of k is more or less arbitrary. Could be tuned via benchmarking.
        // Index will add bitmap.length / k extra space to store the index
        let k = 4; // 25% extra space to store the index
        let blocks = bitmap.length / 32 / k;
        let mut index: Vec<u32> = Vec::with_capacity(blocks);

        // Convert vector of u8 to vector of u32
        let words = div_ceil(bitmap.bitmap.len(), 4);
        let mut bitmap32: Vec<u32> = Vec::with_capacity(words);
        if words > 0 {
            let mut shift = 24;
            let mut word_index = 0;

            bitmap32.push(0);
            for byte in bitmap.bitmap {
                let mut word: u32 = byte.into();
                word <<= shift;
                bitmap32[word_index] |= word;

                if shift == 0 {
                    bitmap32.push(0);
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

        IndexedBitMap {
            length: bitmap.length,
            k: k,
            index: index,
            bitmap: bitmap32,
        }
    }
}

impl IndexedBitMap {
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

    /// Get the index of the nth occurence of 1 in BitMap
    pub fn select(&self, n: usize) -> Option<usize> {
        if n == 0 {
            panic!("select(0)");
        }

        // Use binary search to find block containing nth bit set to 1
        // We can set a lower bound by considering which block would contain n if every bit were
        // set to 1
        let mut low_bound = n / 32 / self.k;
        let mut high_bound = self.index.len();

        if high_bound == 0 {
            // Special case. This bitmap isn't large enough to have an index, so go straight to
            // counting
            return self.select_from_block(0, n);
        }

        let mut block = low_bound + (high_bound - low_bound) / 2;
        loop {
            let count: usize = self.index[block].try_into().unwrap();
            if n == count {
                // Special case. Count at this block is exactly i, so it's in this block, towards
                // the end, so count backwards from the end
                return Some(self.select_first_from_end_of_block(block));
            }

            if n < count {
                // Search earlier blocks
                high_bound = block;
            } else if n > count {
                // Search later blocks
                low_bound = block;
            }
            if high_bound - low_bound == 1 {
                // Search this block
                return self.select_from_block(low_bound, n);
            }
            block = low_bound + (high_bound - low_bound) / 2;
        }
    }

    /// Starting from the end of the given block, search in reverse for the first 1 in the block
    /// and return its position within the BitMap
    fn select_first_from_end_of_block(&self, block_index: usize) -> usize {
        let mut word_index = (block_index + 1) * self.k - 1;
        let mut position = (word_index + 1) * 32;
        let mut word = self.bitmap[word_index];
        let mut mask: u32 = 1;
        while word & mask == 0 {
            if mask == 1 << 31 {
                word_index -= 1;
                word = self.bitmap[word_index];
                mask = 1;
            } else {
                mask <<= 1;
            }
            position -= 1;
        }

        position
    }

    /// Perform a sequential search in the given block for the nth 1 in the array
    fn select_from_block(&self, block_index: usize, n: usize) -> Option<usize> {
        let mut word_index = block_index * self.k;
        let mut position = word_index * 32;
        let mut word = self.bitmap[word_index];
        let mut mask: u32 = 1 << 31;

        let mut count: usize = if block_index > 0 {
            self.index[block_index - 1].try_into().unwrap()
        } else {
            0
        };

        loop {
            if word & mask != 0 {
                count += 1;
                if count == n {
                    return Some(position + 1);
                }
            }

            if mask == 1 {
                word_index += 1;
                if word_index == self.bitmap.len() {
                    return None;
                }
                word = self.bitmap[word_index];
                mask = 1 << 31;
            } else {
                mask >>= 1;
            }
            position += 1;
        }
    }
}

/// Returns n / m with remainder rounded up to nearest integer
fn div_ceil<T>(m: T, n: T) -> T
where
    T: Integer + Copy,
{
    let a = m / n;
    if m % n > zero() {
        a + one()
    } else {
        a
    }
}

#[cfg(test)]
mod tests;
