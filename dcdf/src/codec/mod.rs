//! Encode/Decode Heuristic K^2 Raster
//!
//! An implementation of the algorithm proposed by Silva-Coira, Paramá, de Bernardo, and Seco in
//! their paper, "Space-efficient representations of raster time series".
//!
//! See: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf

use ndarray::{s, Array2, ArrayView2};
use num::{one, zero, Integer, Num, Zero};
use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::Debug;

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
    /// Get the bit at position i
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

/// K^2 raster encoded Snapshot
pub struct Snapshot<T>
where
    T: Num + Copy + PartialOrd + Zero + Debug,
{
    /// Bitmap of tree structure, known as T in Silva-Coira paper
    nodemap: IndexedBitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira paper
    max: Vec<T>,

    /// Tree node minimum values, known as Lmin in Silva-Coira paper
    min: Vec<T>,

    /// The K in K^2 raster.
    k: usize,

    /// Value used to indicate missing value in this snapshot
    nan: T,

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<T> Snapshot<T>
where
    T: Num + Copy + PartialOrd + Zero + Debug,
{
    fn from_array(data: ArrayView2<T>, k: usize, nan: T) -> Self {
        let mut nodemap = BitMap::new();
        let mut max: Vec<T> = vec![];
        let mut min: Vec<T> = vec![];

        let sidelen = data.shape()[0];
        let root = K2TreeNode::from_array(data, k);
        let mut to_traverse = VecDeque::new();
        to_traverse.push_back((root.max, root.min, &root));

        // Breadth first traversal
        while let Some((diff_max, diff_min, child)) = to_traverse.pop_front() {
            max.push(diff_max);

            if !child.children.is_empty() {
                // Non-leaf node
                let elide = child.min == child.max;
                nodemap.push(!elide);
                if !elide {
                    min.push(diff_min);
                    for descendant in &child.children {
                        to_traverse.push_back((
                            child.max - descendant.max,
                            descendant.min - child.min,
                            &descendant,
                        ));
                    }
                }
            }
        }

        let nodemap = IndexedBitMap::from(nodemap);
        Snapshot {
            nodemap,
            max,
            min,
            k,
            nan,
            sidelen,
        }
    }

    /// Get a cell value.
    ///
    /// See: Algorithm 2 in "Scalable and queryable compressed storage structure for raster data" by
    /// Susana Ladra, José R. Paramá, Fernando Silva-Coira, Information Systems 72 (2017) 179-204
    ///
    fn get(&self, row: usize, col: usize) -> T {
        if !self.nodemap.get(0) {
            // Special case, single node tree
            return self.max[0];
        } else {
            self._get(self.sidelen, row, col, 0, self.max[0])
        }
    }

    fn _get(&self, sidelen: usize, row: usize, col: usize, index: usize, max_value: T) -> T {
        let sidelen = sidelen / self.k;
        let index = 1 + self.nodemap.rank(index) * self.k * self.k;
        let index = index + row / sidelen * self.k + col / sidelen;
        let max_value = max_value - self.max[index];

        if index >= self.nodemap.length || !self.nodemap.get(index) {
            // Leaf node
            max_value
        } else {
            // Branch
            self._get(sidelen, row % sidelen, col % sidelen, index, max_value)
        }
    }

    /// Get a subarray of Snapshot
    ///
    /// See: Algorithm 3 in "Scalable and queryable compressed storage structure for raster data"
    /// by Susana Ladra, José R. Paramá, Fernando Silva-Coira, Information Systems 72 (2017)
    /// 179-204
    ///
    /// This is based on that algorithm, but has been modified to return a submatrix rather than an
    /// unordered sequence of values.
    ///
    fn get_window(&self, top: usize, bottom: usize, left: usize, right: usize) -> Array2<T> {
        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);
        self._get_window(
            self.sidelen,
            top,
            bottom - 1,
            left,
            right - 1,
            0,
            self.max[0],
            &mut window,
            top,
            left,
            0,
            0,
        );

        window
    }

    fn _get_window(
        &self,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index: usize,
        max_value: T,
        window: &mut Array2<T>,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
    ) {
        let sidelen = sidelen / self.k;
        let index = 1 + self.nodemap.rank(index) * self.k * self.k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min((i + 1) * sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min((j + 1) * sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * self.k + j;
                let max_value_ = max_value - self.max[index_];

                if index_ >= self.nodemap.length || !self.nodemap.get(index_) {
                    // Leaf node
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            window[[
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                            ]] = max_value_;
                        }
                    }
                } else {
                    // Branch
                    self._get_window(
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_,
                        max_value_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                    );
                }
            }
        }
    }

    /// Search the window for cells with values in a given range
    ///
    /// See: Algorithm 4 in "Scalable and queryable compressed storage structure for raster data"
    /// by Susana Ladra, José R. Paramá, Fernando Silva-Coira, Information Systems 72 (2017)
    /// 179-204
    ///
    fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: T,
        upper: T,
    ) -> Vec<(usize, usize)> {
        let mut cells: Vec<(usize, usize)> = vec![];

        self._search_window(
            self.sidelen,
            top,
            bottom - 1,
            left,
            right - 1,
            lower,
            upper,
            0,
            self.min[0],
            self.max[0],
            &mut cells,
            0,
            0,
        );

        cells
    }

    fn _search_window(
        &self,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: T,
        upper: T,
        index: usize,
        min_value: T,
        max_value: T,
        cells: &mut Vec<(usize, usize)>,
        top_offset: usize,
        left_offset: usize,
    ) {
        let sidelen = sidelen / self.k;
        let index = 1 + self.nodemap.rank(index) * self.k * self.k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min((i + 1) * sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min((j + 1) * sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * self.k + j;
                let max_value_ = max_value - self.max[index_];

                if index_ >= self.nodemap.length || !self.nodemap.get(index_) {
                    // Leaf node
                    if lower <= max_value_ && max_value_ <= upper {
                        for row in top_..=bottom_ {
                            for col in left_..=right_ {
                                cells.push((top_offset_ + row, left_offset_ + col));
                            }
                        }
                    }
                } else {
                    // Branch
                    let min_value_ = min_value + self.min[self.nodemap.rank(index_)];
                    if lower <= min_value && max_value_ <= upper {
                        // All values in branch are within bounds
                        for row in top_..=bottom_ {
                            for col in left_..=right_ {
                                cells.push((top_offset_ + row, left_offset_ + col));
                            }
                        }
                    } else if upper >= min_value_ || lower <= max_value_ {
                        // Some, but not all, values in branch are within bounds.
                        // Recurse into branch
                        self._search_window(
                            sidelen,
                            top_,
                            bottom_,
                            left_,
                            right_,
                            lower,
                            upper,
                            index_,
                            min_value_,
                            max_value_,
                            cells,
                            top_offset_,
                            left_offset_,
                        );
                    }
                }
            }
        }
    }
}

// Temporary tree structure for building K^2 raster
struct K2TreeNode<T>
where
    T: Num + Copy + PartialOrd + Zero + Debug,
{
    max: T,
    min: T,
    children: Vec<K2TreeNode<T>>,
}

impl<T> K2TreeNode<T>
where
    T: Num + Copy + PartialOrd + Zero + Debug,
{
    fn from_array(data: ArrayView2<T>, k: usize) -> Self {
        let sidelen = data.shape()[0];

        // Leaf node
        if sidelen == 1 {
            return K2TreeNode {
                max: data[[0, 0]],
                min: data[[0, 0]],
                children: vec![],
            };
        }

        // Branch
        let mut children: Vec<K2TreeNode<T>> = vec![];
        let step = sidelen / k;
        for row in 0..k {
            for col in 0..k {
                let branch = data.slice(s![
                    row * step..(row + 1) * step,
                    col * step..(col + 1) * step
                ]);
                children.push(K2TreeNode::from_array(branch, k));
            }
        }

        let mut max = children[0].max;
        let mut min = children[0].min;
        for child in &children[1..] {
            if child.max > max {
                max = child.max;
            }
            if child.min < min {
                min = child.min;
            }
        }

        K2TreeNode { min, max, children }
    }
}

pub fn encode_snapshot<T>(data: ArrayView2<T>) -> Vec<u8> {
    b"booty".iter().cloned().collect()
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
