//! Encode/Decode Heuristic K^2 Raster
//!
//! An implementation of the algorithm proposed by Silva-Coira, Paramá, de Bernardo, and Seco in
//! their paper, "Space-efficient representations of raster time series".
//!
//! See: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf

use ndarray::{Array2, ArrayView2};
use num_traits::{AsPrimitive, PrimInt};
use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::mem::size_of;

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
struct BitMap {
    length: usize,
    bitmap: Vec<u8>,
}

impl BitMap {
    /// Initialize an empty BitMap
    fn new() -> BitMap {
        BitMap {
            length: 0,
            bitmap: vec![],
        }
    }

    /// Push a bit onto the BitMap
    fn push(&mut self, bit: bool) {
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
struct IndexedBitMap {
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
            k,
            index,
            bitmap: bitmap32,
        }
    }
}

impl IndexedBitMap {
    /// Get the bit at position i
    fn get(&self, i: usize) -> bool {
        let word_index = i / 32;
        let bit_index = i % 32;
        let shift = 31 - bit_index;
        let word = self.bitmap[word_index];

        (word >> shift) & 1 > 0
    }

    /// Count occurences of 1 in BitMap[0...i]
    fn rank(&self, i: usize) -> usize {
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
    fn rank0(&self, i: usize) -> usize {
        i - self.rank(i)
    }

    /// Get the index of the nth occurence of 1 in BitMap
    fn select(&self, n: usize) -> Option<usize> {
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

/// Compact storage for integers (Directly Addressable Codes)
struct Dacs {
    levels: Vec<(IndexedBitMap, Vec<u8>)>,
}

impl Dacs {
    fn get<T>(&self, index: usize) -> T
    where
        T: PrimInt + Debug,
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

        let n: T = T::from(zigzag_decode(n)).unwrap();
        n
    }
}

impl<T> From<Vec<T>> for Dacs
where
    T: PrimInt + Debug,
{
    fn from(data: Vec<T>) -> Self {
        // Set up levels. Probably won't need all of them
        let mut levels = Vec::with_capacity(8);
        for _ in 0..8 {
            levels.push((BitMap::new(), Vec::new()));
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
            .take_while(|(bitmap, bytes)| bitmap.length > 0)
            .map(|(bitmap, bytes)| (IndexedBitMap::from(bitmap), bytes))
            .collect();

        Dacs { levels }
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

/// K^2 raster encoded Snapshot
pub struct Snapshot {
    /// Bitmap of tree structure, known as T in Silva-Coira paper
    nodemap: IndexedBitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira paper
    max: Dacs,

    /// Tree node minimum values, known as Lmin in Silva-Coira paper
    min: Dacs,

    /// The K in K^2 raster.
    k: i32,

    /// Shape of the encoded raster. Since K^2 matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl Snapshot {
    pub fn from_array<T>(data: ArrayView2<T>, k: i32) -> Self
    where
        T: PrimInt + Debug,
    {
        let mut nodemap = BitMap::new();
        let mut max: Vec<T> = vec![];
        let mut min: Vec<T> = vec![];

        // Compute the smallest square with sides whose length is a power of K that will contain
        // the passed in data.
        let shape = data.shape();
        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let root = K2TreeNode::from_array(data, k, sidelen);
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

        Snapshot {
            nodemap: IndexedBitMap::from(nodemap),
            max: Dacs::from(max),
            min: Dacs::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value.
    ///
    /// See: Algorithm 2 in "Scalable and queryable compressed storage structure for raster data" by
    /// Susana Ladra, José R. Paramá, Fernando Silva-Coira, Information Systems 72 (2017) 179-204
    ///
    pub fn get<T>(&self, row: usize, col: usize) -> T
    where
        T: PrimInt + Debug,
    {
        self.check_bounds(row, col);

        if !self.nodemap.get(0) {
            // Special case, single node tree
            return self.max.get(0);
        } else {
            self._get(self.sidelen, row, col, 0, self.max.get(0))
        }
    }

    fn _get<T>(&self, sidelen: usize, row: usize, col: usize, index: usize, max_value: T) -> T
    where
        T: PrimInt + Debug,
    {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;
        let index = index + row / sidelen * k + col / sidelen;
        let max_value = max_value - self.max.get(index);

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
    pub fn get_window<T>(&self, top: usize, bottom: usize, left: usize, right: usize) -> Array2<T>
    where
        T: PrimInt + Debug,
    {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom, right);

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
            self.max.get(0),
            &mut window,
            top,
            left,
            0,
            0,
        );

        window
    }

    fn _get_window<T>(
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
    ) where
        T: PrimInt + Debug,
    {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * k + j;
                let max_value_ = max_value - self.max.get(index_);

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
    pub fn search_window<T>(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: T,
        upper: T,
    ) -> Vec<(usize, usize)>
    where
        T: PrimInt + Debug,
    {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom, right);

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
            self.min.get(0),
            self.max.get(0),
            &mut cells,
            0,
            0,
        );

        cells
    }

    fn _search_window<T>(
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
    ) where
        T: PrimInt + Debug,
    {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * k + j;
                let max_value_ = max_value - self.max.get(index_);

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
                    let min_value_ = min_value + self.min.get(self.nodemap.rank(index_));
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

    /// Panics if given point is out of bounds for this snapshot
    fn check_bounds(&self, row: usize, col: usize) {
        if row >= self.shape[0] || col >= self.shape[1] {
            panic!(
                "dcdf::Snapshot: index[{}, {}] is out of bounds for array of shape {:?}",
                row, col, self.shape
            );
        }
    }
}

// Temporary tree structure for building K^2 raster
struct K2TreeNode<T>
where
    T: PrimInt + Debug,
{
    max: T,
    min: T,
    children: Vec<K2TreeNode<T>>,
}

impl<T> K2TreeNode<T>
where
    T: PrimInt + Debug,
{
    fn from_array(data: ArrayView2<T>, k: i32, sidelen: usize) -> Self {
        Self::_from_array(data, k as usize, sidelen, 0, 0)
    }

    fn _from_array(data: ArrayView2<T>, k: usize, sidelen: usize, row: usize, col: usize) -> Self {
        // Leaf node
        if sidelen == 1 {
            // Fill cells that lay outside of original raster with 0s
            let shape = data.shape();
            let rows = shape[0];
            let cols = shape[1];
            let value = if row < rows && col < cols {
                data[[row, col]]
            } else {
                T::zero()
            };
            return K2TreeNode {
                max: value,
                min: value,
                children: vec![],
            };
        }

        // Branch
        let mut children: Vec<K2TreeNode<T>> = vec![];
        let sidelen = sidelen / k;
        for i in 0..k {
            let row_ = row + i * sidelen;
            for j in 0..k {
                let col_ = col + j * sidelen;
                children.push(K2TreeNode::_from_array(data, k, sidelen, row_, col_));
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

/// T - K^2 Raster Log
struct Log {
    /// Bitmap of tree structure, known as T in Silva-Coira paper
    nodemap: IndexedBitMap,

    /// Bit map of tree nodes that match referenced snapshot, or have cells that all differ by the
    /// same amount, known as eqB in Silva-Coira paper
    equal: IndexedBitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira paper
    max: Dacs,

    /// Tree node minimum values, known as Lmin in Silva-Coira paper
    min: Dacs,

    /// The K in K^2 raster.
    k: i32,

    /// Shape of the encoded raster. Since K^2 matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl Log {
    fn from_arrays<T>(snapshot: ArrayView2<T>, log: ArrayView2<T>, k: i32) -> Self
    where
        T: PrimInt + Debug,
    {
        let mut nodemap = BitMap::new();
        let mut equal = BitMap::new();
        let mut max: Vec<i64> = vec![];
        let mut min: Vec<i64> = vec![];

        // Compute the smallest square with sides whose length is a power of K that will contain
        // the passed in data.
        let shape = snapshot.shape();
        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let root = K2PTreeNode::from_arrays(snapshot, log, k, sidelen);
        let mut to_traverse = VecDeque::new();
        to_traverse.push_back(&root);

        // Breadth first traversal
        while let Some(node) = to_traverse.pop_front() {
            let max_t = node.max_t.to_i64().unwrap();
            let max_s = node.max_s.to_i64().unwrap();
            max.push(max_t - max_s);

            if !node.children.is_empty() {
                // Non-leaf node
                if node.min_t == node.max_t {
                    // Log quadbox is uniform, terminate here
                    nodemap.push(false);
                    equal.push(false);
                } else if node.equal {
                    // Difference of log and snapshot quadboxes is uniform, terminate here
                    nodemap.push(false);
                    equal.push(true);
                } else {
                    // Regular old internal node, keep going
                    nodemap.push(true);
                    let min_t = node.min_t.to_i64().unwrap();
                    let min_s = node.min_s.to_i64().unwrap();
                    min.push(min_t - min_s);
                    for child in &node.children {
                        to_traverse.push_back(child);
                    }
                }
            }
        }

        Log {
            nodemap: IndexedBitMap::from(nodemap),
            equal: IndexedBitMap::from(equal),
            max: Dacs::from(max),
            min: Dacs::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value
    ///
    /// See: Algorithm 3 in Silva-Coira paper
    ///
    pub fn get<T>(&self, snapshot: &Snapshot, row: usize, col: usize) -> T
    where
        T: PrimInt + Debug,
    {
        self.check_bounds(row, col);

        let max_t: T = self.max.get(0);
        let max_s: T = snapshot.max.get(0);
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);
        if single_t && single_s {
            // Both trees have single node
            max_t + max_s
        } else if single_t && !self.equal.get(0) {
            // Log has single node but it contains a uniform value for all cells
            max_t + max_s
        } else {
            let index_t = if single_t { None } else { Some(0) };
            let index_s = if single_s { None } else { Some(0) };
            self._get(
                snapshot,
                self.sidelen,
                row,
                col,
                index_t,
                index_s,
                max_t,
                max_s,
            )
        }
    }

    fn _get<T>(
        &self,
        snapshot: &Snapshot,
        sidelen: usize,
        row: usize,
        col: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: T,
        max_s: T,
    ) -> T
    where
        T: PrimInt + Debug,
    {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let mut max_s = max_s;
        let mut max_t = max_t;

        let index_s = if let Some(index) = index_s {
            let index = 1 + snapshot.nodemap.rank(index) * k * k;
            let index = index + row / sidelen * k + col / sidelen;
            max_s = max_s - snapshot.max.get(index);
            Some(index)
        } else {
            None
        };

        let index_t = if let Some(index) = index_t {
            let index = 1 + self.nodemap.rank(index) * k * k;
            let index = index + row / sidelen * k + col / sidelen;
            max_t = self.max.get(index);
            Some(index)
        } else {
            None
        };

        let leaf_t = if let Some(index) = index_t {
            index > self.nodemap.length || !self.nodemap.get(index)
        } else {
            true
        };

        let leaf_s = if let Some(index) = index_s {
            index > snapshot.nodemap.length || !snapshot.nodemap.get(index)
        } else {
            true
        };

        if leaf_t && leaf_s {
            max_t + max_s
        } else if leaf_s {
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                index_t,
                None,
                max_t,
                max_s,
            )
        } else if leaf_t {
            if let Some(index_t) = index_t {
                if index_t < self.nodemap.length {
                    let equal = self.equal.get(self.nodemap.rank0(index_t + 1) - 1);
                    if !equal {
                        return max_t + max_s;
                    }
                }
            }
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                None,
                index_s,
                max_t,
                max_s,
            )
        } else {
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                index_t,
                index_s,
                max_t,
                max_s,
            )
        }
    }

    /// Get a subarray of log
    ///
    /// See: Algorithm 3 in "Scalable and queryable compressed storage structure for raster data"
    /// by Susana Ladra, José R. Paramá, Fernando Silva-Coira, Information Systems 72 (2017)
    /// 179-204
    ///
    /// This is based on that algorithm, but has been modified to return a submatrix rather than an
    /// unordered sequence of values.
    ///
    pub fn get_window<T>(
        &self,
        snapshot: &Snapshot,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array2<T>
    where
        T: PrimInt + Debug,
    {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom, right);

        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);

        self._get_window(
            snapshot,
            self.sidelen,
            top,
            bottom - 1,
            left,
            right - 1,
            Some(0),
            Some(0),
            self.max.get(0),
            snapshot.max.get(0),
            &mut window,
            top,
            left,
            0,
            0,
            0,
        );

        window
    }

    fn _get_window<T>(
        &self,
        snapshot: &Snapshot,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: T,
        max_s: T,
        window: &mut Array2<T>,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
        level: usize,
    ) where
        T: PrimInt + Debug,
    {
        let mut indent = String::new();
        for _ in 0..level {
            indent.push_str("    ");
        }

        println!(
            "{indent}himom: {sidelen} {top}..{bottom} {left}..{right} {top_offset} {left_offset}"
        );
        let k = self.k as usize;
        let sidelen = sidelen / k;

        let index_t = match index_t {
            Some(index) => Some(1 + self.nodemap.rank(index) * k * k),
            None => None,
        };

        let index_s = match index_s {
            Some(index) => Some(1 + snapshot.nodemap.rank(index) * k * k),
            None => None,
        };

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                println!("{indent}i={i}, j={j}");
                let index_t_ = match index_t {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_t_ = match index_t_ {
                    Some(index) => self.max.get(index),
                    None => max_t,
                };

                let leaf_t = match index_t_ {
                    Some(index) => index > self.nodemap.length || !self.nodemap.get(index),
                    None => true,
                };

                let index_s_ = match index_s {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_s_ = match index_s_ {
                    Some(index) => max_s - snapshot.max.get(index),
                    None => max_s,
                };

                let leaf_s = match index_s_ {
                    Some(index) => index > snapshot.nodemap.length || !snapshot.nodemap.get(index),
                    None => true,
                };

                if leaf_t && leaf_s {
                    let value = max_t_ + max_s_;
                    println!("{indent}two leaves: {value:?}");
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            println!(
                                "{indent}set ({}, {})",
                                top_offset_ + row,
                                left_offset_ + col
                            );
                            window[[
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                            ]] = value;
                        }
                    }
                } else if leaf_s {
                    println!("{indent}leaf s");
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_t_,
                        None,
                        max_t_,
                        max_s_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                        level + 1,
                    );
                } else if leaf_t {
                    println!("{indent}leaf t");
                    if let Some(index) = index_t_ {
                        if !self.nodemap.get(index) {
                            let equal = self.equal.get(self.nodemap.rank0(index + 1) - 1);
                            if !equal {
                                let value = max_t_ + max_s_;
                                println!("{indent}uniform: {value:?}");
                                for row in top_..=bottom_ {
                                    for col in left_..=right_ {
                                        println!(
                                            "{indent}set ({}, {})",
                                            top_offset_ + row,
                                            left_offset_ + col
                                        );
                                        window[[
                                            top_offset_ + row - window_top,
                                            left_offset_ + col - window_left,
                                        ]] = value;
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        None,
                        index_s_,
                        max_t_,
                        max_s_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                        level + 1,
                    );
                } else {
                    println!("{indent}no leaves");
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_t_,
                        index_s_,
                        max_t_,
                        max_s_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                        level + 1,
                    );
                }
            }
        }
    }

    /// Panics if given point is out of bounds for this log
    fn check_bounds(&self, row: usize, col: usize) {
        if row >= self.shape[0] || col >= self.shape[1] {
            panic!(
                "dcdf::Log: index[{}, {}] is out of bounds for array of shape {:?}",
                row, col, self.shape
            );
        }
    }
}

// Temporary tree structure for building T - K^2 raster
struct K2PTreeNode<T>
where
    T: PrimInt + Debug,
{
    max_t: T,
    min_t: T,
    max_s: T,
    min_s: T,
    diff: i64,
    equal: bool,
    children: Vec<K2PTreeNode<T>>,
}

impl<T> K2PTreeNode<T>
where
    T: PrimInt + Debug,
{
    fn from_arrays(snapshot: ArrayView2<T>, log: ArrayView2<T>, k: i32, sidelen: usize) -> Self {
        Self::_from_arrays(snapshot, log, k as usize, sidelen, 0, 0)
    }

    fn _from_arrays(
        snapshot: ArrayView2<T>,
        log: ArrayView2<T>,
        k: usize,
        sidelen: usize,
        row: usize,
        col: usize,
    ) -> Self {
        // Leaf node
        if sidelen == 1 {
            // Fill cells that lay outside of original raster with 0s
            let shape = snapshot.shape();
            let rows = shape[0];
            let cols = shape[1];
            let value_s = if row < rows && col < cols {
                snapshot[[row, col]]
            } else {
                T::zero()
            };
            let value_t = if row < rows && col < cols {
                log[[row, col]]
            } else {
                T::zero()
            };
            let diff = value_t.to_i64().unwrap() - value_s.to_i64().unwrap();
            return K2PTreeNode {
                max_t: value_t,
                min_t: value_t,
                max_s: value_s,
                min_s: value_s,
                diff: diff,
                equal: true,
                children: vec![],
            };
        }

        // Branch
        let mut children: Vec<K2PTreeNode<T>> = vec![];
        let sidelen = sidelen / k;
        for i in 0..k {
            let row_ = row + i * sidelen;
            for j in 0..k {
                let col_ = col + j * sidelen;
                children.push(K2PTreeNode::_from_arrays(
                    snapshot, log, k, sidelen, row_, col_,
                ));
            }
        }

        let mut max_t = children[0].max_t;
        let mut min_t = children[0].min_t;
        let mut max_s = children[0].max_s;
        let mut min_s = children[0].min_s;
        let mut equal = children.iter().all(|child| child.equal);
        let diff = children[0].diff;
        for child in &children[1..] {
            if child.max_t > max_t {
                max_t = child.max_t;
            }
            if child.min_t < min_t {
                min_t = child.min_t;
            }
            if child.max_s > max_s {
                max_s = child.max_s;
            }
            if child.min_s < min_s {
                min_s = child.min_s;
            }
            equal = equal && child.diff == diff;
        }

        K2PTreeNode {
            min_t,
            max_t,
            min_s,
            max_s,
            diff,
            equal,
            children,
        }
    }
}

/// Returns n / m with remainder rounded up to nearest integer
fn div_ceil<T>(m: T, n: T) -> T
where
    T: PrimInt + Debug,
{
    let a = m / n;
    if m % n > T::zero() {
        a + T::one()
    } else {
        a
    }
}

/// Make sure bounds are ordered correctly, eg right is to the right of left, top is above
/// bottom.
fn rearrange(lower: usize, upper: usize) -> (usize, usize) {
    if lower > upper {
        (upper, lower)
    } else {
        (lower, upper)
    }
}

#[cfg(test)]
mod tests;
