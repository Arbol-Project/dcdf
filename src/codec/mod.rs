//! Encode/Decode Heuristic K^2 Raster
//!
//! An implementation of the algorithm proposed by Silva-Coira, Paramá, de Bernardo, and
//! Seco in their paper, "Space-efficient representations of raster time series".
//!
//! See: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf

// Remove later. Currently all private methods that are only used in tests so far are
// considered dead code. They will get used eventually, but the warnings are annoying
// at this stage of development
#![allow(dead_code)]

/// A sequence of bit values (0 or 1)
///
struct BitMap {
    length: usize,
    packed: Vec<u8>,
}

// This is a very naive implementation. By adding a little bit extra data can vastly
// improve performance for rank and select queries. Will be revised shortly to use
// something more performant.
//
// See: https://users.dcc.uchile.cl/~gnavarro/algoritmos/ps/wea05.pdf
//
impl BitMap {
    /// Construct an empty BitMap
    fn new() -> BitMap {
        BitMap {
            length: 0,
            packed: vec![],
        }
    }

    /// Pushes a bit onto the BitMap
    fn push(&mut self, bit: bool) {
        // Which bit do we need to set in the relevant byte?
        let position = self.length % 8;

        // How much do we need to shift to the left to get to that position?
        let shift = 7 - position;

        // If position == 0, we start a new byte
        if position == 0 {
            self.packed.push(if bit { 1 << shift } else { 0 });
        }
        // Otherwise add bit to currently started byte
        else if bit {
            let last = self.packed.len() - 1;
            self.packed[last] += 1 << shift;
        }

        self.length += 1;
    }

    fn iter(&self) -> BitMapIterator {
        BitMapIterator::new(&self)
    }

    /// Counts occurence of 1 in BitMap[0...i]
    fn rank1(&self, i: usize) -> usize {
        if i > self.length - 1 {
            // Can only happen if there is a programming error in this module
            panic!("index out of bounds. length: {}, i: {}", self.length, i);
        }

        let mut count = 0;
        for bit in self.iter().take(i + 1) {
            if bit {
                count += 1;
            }
        }

        count
    }

    /// Returns index of nth occurence of 1 in BitMap
    fn select1(&self, n: usize) -> usize {
        let mut count = 0;

        for (i, bit) in self.iter().enumerate() {
            if bit {
                count += 1;
                println!("hello? n={} i={} bit={} count={}", n, i, bit, count);

                if count == n {
                    return i;
                }
            }
        }

        // Can only happen if there is a programming error in this module
        panic!("select out of bounds. count: {}, n: {}", count, n);
    }
}

struct BitMapIterator<'a> {
    bitmap: &'a BitMap,
    position: usize,
    mask: u8,
}

impl BitMapIterator<'_> {
    fn new(bitmap: &BitMap) -> BitMapIterator {
        BitMapIterator {
            bitmap: bitmap,
            position: 0,
            mask: 1 << 7,
        }
    }
}

impl Iterator for BitMapIterator<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.bitmap.length {
            return None;
        }

        let index = self.position / 8;
        let bit = self.bitmap.packed[index] & self.mask;

        self.position += 1;
        if self.mask == 1 {
            self.mask = 1 << 7;
        } else {
            self.mask >>= 1;
        }

        Some(bit != 0)
    }
}

#[cfg(test)]
mod tests;
