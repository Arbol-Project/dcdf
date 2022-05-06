//! Encode/Decode Heuristic K²-Raster
//!
//! An implementation of the compact data structure proposed by Silva-Coira, et al.[^bib1],
//! which, in turn, is based on work by Ladra[^bib2] and González[^bib3].
//!
//! The data structures here provide a means of storing raster data compactly while still being
//! able to run queries in-place on the stored data. A separate decompression step is not required
//! in order to read the data.
//!
//! For insight into how this data structure works, please see the literature in footnotes.
//! Reproducing the literature is outside of the scope for this documentation.
//!
//! [^bib1]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient representations
//!     of raster time series, Information Sciences 566 (2021) 300-325.][1]
//!
//! [^bib2]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
//!     structure for raster data, Information Systems 72 (2017) 179-204.
//!
//! [^bib3]: [F. González, S. Grabowski, V. Mäkinen, G. Navarro, Practical implementations of rank
//!     and select queries, in: Poster Proc. of 4th Workshop on Efficient and Experimental
//!     Algorithms (WEA) Greece, 2005, pp. 27-38.][2]
//!
//! [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
//! [2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9548&rep=rep1&type=pdf

use ndarray::{Array2, ArrayView2};
use num_traits::PrimInt;
use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;

/// A short series of time instants made up of one Snapshot encoding the first time instant and
/// Logs encoding subsequent time instants.
///
struct Block<T> {
    _marker: PhantomData<T>,

    /// Snapshot of first time instant
    snapshot: Snapshot<T>,

    /// Successive time instants as logs
    logs: Vec<Log<T>>,
}

impl<T> Block<T> {
    fn new(snapshot: Snapshot<T>, logs: Vec<Log<T>>) -> Self {
        Self {
            _marker: PhantomData,
            snapshot: snapshot,
            logs: logs,
        }
    }

    fn get(&self, instant: usize, row: usize, col: usize) -> T
    where
        T: PrimInt + Debug,
    {
        match instant {
            0 => self.snapshot.get(row, col),
            _ => self.logs[instant - 1].get(&self.snapshot, row, col),
        }
    }

    fn get_window(
        &self,
        instant: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array2<T>
    where
        T: PrimInt + Debug,
    {
        match instant {
            0 => self.snapshot.get_window(top, bottom, left, right),
            _ => self.logs[instant - 1].get_window(&self.snapshot, top, bottom, left, right),
        }
    }

    pub fn search_window(
        &self,
        instant: usize,
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
        match instant {
            0 => self
                .snapshot
                .search_window(top, bottom, left, right, lower, upper),
            _ => self.logs[instant - 1].search_window(
                &self.snapshot,
                top,
                bottom,
                left,
                right,
                lower,
                upper,
            ),
        }
    }
}

/// K²-Raster encoded Snapshot
///
/// A Snapshot stores raster data for a particular time instant in a raster time series. Data is
/// stored standalone without reference to any other time instant.
///
pub struct Snapshot<T> {
    _marker: PhantomData<T>,

    /// Bitmap of tree structure, known as T in Silva-Coira
    nodemap: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    max: Dacs,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    min: Dacs,

    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<T> Snapshot<T>
where
    T: PrimInt + Debug,
{
    /// Build a snapshot from a two-dimensional array.
    ///
    pub fn from_array(data: ArrayView2<T>, k: i32) -> Self {
        let mut nodemap = BitMapBuilder::new();
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
            _marker: PhantomData,
            nodemap: BitMap::from(nodemap),
            max: Dacs::from(max),
            min: Dacs::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value.
    ///
    /// See: Algorithm 2 in Ladra[^note]
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn get(&self, row: usize, col: usize) -> T {
        self.check_bounds(row, col);

        if !self.nodemap.get(0) {
            // Special case, single node tree
            return self.max.get(0);
        } else {
            self._get(self.sidelen, row, col, 0, self.max.get(0))
        }
    }

    fn _get(&self, sidelen: usize, row: usize, col: usize, index: usize, max_value: T) -> T {
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
    /// This is based on Algorithm 3 in Ladra[^note], but has been modified to return a submatrix
    /// rather than an unordered sequence of values.
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn get_window(&self, top: usize, bottom: usize, left: usize, right: usize) -> Array2<T> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom, right);

        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value = self.max.get(0);
            for row in 0..rows {
                for col in 0..cols {
                    window[[row, col]] = value;
                }
            }
        } else {
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
        }

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
    /// See: Algorithm 4 in Ladra[^note]
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: T,
        upper: T,
    ) -> Vec<(usize, usize)> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bottom, right);

        let mut cells: Vec<(usize, usize)> = vec![];

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value: T = self.max.get(0);
            if lower <= value && value <= upper {
                for row in top..bottom {
                    for col in left..right {
                        cells.push((row, col));
                    }
                }
            }
        } else {
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
        }

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

/// Temporary tree structure for building K^2 raster
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

/// K²-Raster encoded Log
///
/// A Log stores raster data for a particular time instant in a raster time series as the
/// difference between this time instant and a reference Snapshot.
///
pub struct Log<T> {
    _marker: PhantomData<T>,

    /// Bitmap of tree structure, known as T in Silva-Coira
    nodemap: BitMap,

    /// Bitmap of tree nodes that match referenced snapshot, or have cells that all differ by the
    /// same amount, known as eqB in Silva-Coira paper
    equal: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    max: Dacs,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    min: Dacs,

    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<T> Log<T>
where
    T: PrimInt + Debug,
{
    /// Build a snapshot from a pair of two-dimensional arrays.
    ///
    pub fn from_arrays(snapshot: ArrayView2<T>, log: ArrayView2<T>, k: i32) -> Self {
        let mut nodemap = BitMapBuilder::new();
        let mut equal = BitMapBuilder::new();
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
            _marker: PhantomData,
            nodemap: BitMap::from(nodemap),
            equal: BitMap::from(equal),
            max: Dacs::from(max),
            min: Dacs::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value
    ///
    /// See: Algorithm 3 in Silva-Coira[^note]
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn get(&self, snapshot: &Snapshot<T>, row: usize, col: usize) -> T {
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
            let value = self._get(
                snapshot,
                self.sidelen,
                row,
                col,
                index_t,
                index_s,
                max_t.to_i64().unwrap(),
                max_s.to_i64().unwrap(),
            );

            T::from(value).unwrap()
        }
    }

    fn _get(
        &self,
        snapshot: &Snapshot<T>,
        sidelen: usize,
        row: usize,
        col: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: i64,
        max_s: i64,
    ) -> i64 {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let mut max_s = max_s;
        let mut max_t = max_t;

        let index_s = match index_s {
            Some(index) => {
                let index = 1 + snapshot.nodemap.rank(index) * k * k;
                let index = index + row / sidelen * k + col / sidelen;
                max_s = max_s - snapshot.max.get::<i64>(index);
                Some(index)
            }
            None => None,
        };

        let index_t = match index_t {
            Some(index) => {
                let index = 1 + self.nodemap.rank(index) * k * k;
                let index = index + row / sidelen * k + col / sidelen;
                max_t = self.max.get(index);
                Some(index)
            }
            None => None,
        };

        let leaf_t = match index_t {
            Some(index) => index > self.nodemap.length || !self.nodemap.get(index),
            None => true,
        };

        let leaf_s = match index_s {
            Some(index) => index > snapshot.nodemap.length || !snapshot.nodemap.get(index),
            None => true,
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
    /// This is based on Algorithm 5 in Silva-Coira[^note], but has been modified to return a
    /// submatrix rather than an unordered sequence of values.
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn get_window(
        &self,
        snapshot: &Snapshot<T>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array2<T> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom, right);

        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);

        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        if single_t && (single_s || !self.equal.get(0)) {
            // Both trees have single node or log has single node but it contains a uniform value
            // for all cells
            let max_t: T = self.max.get(0);
            let max_s: T = snapshot.max.get(0);
            for row in 0..rows {
                for col in 0..cols {
                    window[[row, col]] = max_t + max_s;
                }
            }
        } else {
            self._get_window(
                snapshot,
                self.sidelen,
                top,
                bottom - 1,
                left,
                right - 1,
                if single_t { None } else { Some(0) },
                if single_s { None } else { Some(0) },
                self.max.get(0),
                snapshot.max.get(0),
                &mut window,
                top,
                left,
                0,
                0,
            );
        }

        window
    }

    fn _get_window(
        &self,
        snapshot: &Snapshot<T>,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: i64,
        max_s: i64,
        window: &mut Array2<T>,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
    ) {
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
                    Some(index) => max_s - snapshot.max.get::<i64>(index),
                    None => max_s,
                };

                let leaf_s = match index_s_ {
                    Some(index) => index > snapshot.nodemap.length || !snapshot.nodemap.get(index),
                    None => true,
                };

                if leaf_t && leaf_s {
                    let value = max_t_ + max_s_;
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            window[[
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                            ]] = T::from(value).unwrap();
                        }
                    }
                } else if leaf_s {
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
                    );
                } else if leaf_t {
                    if let Some(index) = index_t_ {
                        if !self.nodemap.get(index) {
                            let equal = self.equal.get(self.nodemap.rank0(index + 1) - 1);
                            if !equal {
                                let value = max_t_ + max_s_;
                                for row in top_..=bottom_ {
                                    for col in left_..=right_ {
                                        window[[
                                            top_offset_ + row - window_top,
                                            left_offset_ + col - window_left,
                                        ]] = T::from(value).unwrap();
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
                    );
                } else {
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
                    );
                }
            }
        }
    }

    /// Search the window for cells with values in a given range
    ///
    /// See: Algorithm 7 in Silva-Coira[^note]
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn search_window(
        &self,
        snapshot: &Snapshot<T>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: T,
        upper: T,
    ) -> Vec<(usize, usize)> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bottom, right);

        let mut cells: Vec<(usize, usize)> = vec![];
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        self._search_window(
            snapshot,
            self.sidelen,
            top,
            bottom - 1,
            left,
            right - 1,
            lower.to_i64().unwrap(),
            upper.to_i64().unwrap(),
            if single_t { None } else { Some(0) },
            if single_s { None } else { Some(0) },
            self.min.get(0),
            snapshot.min.get(0),
            self.max.get(0),
            snapshot.max.get(0),
            &mut cells,
            0,
            0,
        );

        cells
    }

    fn _search_window(
        &self,
        snapshot: &Snapshot<T>,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i64,
        upper: i64,
        index_t: Option<usize>,
        index_s: Option<usize>,
        min_t: i64,
        min_s: i64,
        max_t: i64,
        max_s: i64,
        cells: &mut Vec<(usize, usize)>,
        top_offset: usize,
        left_offset: usize,
    ) {
        let max_value = max_s + max_t;
        let min_value = min_s + min_t;

        if min_value >= lower && max_value <= upper {
            // Branch meets condition, output all cells
            for row in top..=bottom {
                for col in left..=right {
                    cells.push((top_offset + row, left_offset + col));
                }
            }
            return;
        } else if min_value > upper || max_value < lower {
            // No cells in this branch meet the condition
            return;
        }

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

                let mut index_t_ = match index_t {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let mut index_s_ = match index_s {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_t_ = match index_t_ {
                    Some(index) => self.max.get(index),
                    None => max_t,
                };

                let max_s_ = match index_s_ {
                    Some(index) => max_s - snapshot.max.get::<i64>(index),
                    None => max_s,
                };

                let leaf_t = match index_t_ {
                    Some(index) => index >= self.nodemap.length || !self.nodemap.get(index),
                    None => true,
                };

                let leaf_s = match index_s_ {
                    Some(index) => index >= snapshot.nodemap.length || !snapshot.nodemap.get(index),
                    None => true,
                };

                let mut min_t_ = match index_t_ {
                    Some(index) => {
                        if leaf_t {
                            min_t
                        } else {
                            self.min.get(self.nodemap.rank(index))
                        }
                    }
                    None => min_t,
                };

                let mut min_s_ = match index_s_ {
                    Some(index) => {
                        if leaf_s {
                            min_s
                        } else {
                            min_s + snapshot.min.get::<i64>(snapshot.nodemap.rank(index))
                        }
                    }
                    None => min_s,
                };

                if leaf_s {
                    min_s_ = max_s_;
                    index_s_ = None;
                }

                if leaf_t {
                    min_t_ = max_t_;
                    if let Some(index) = index_t_ {
                        if index < self.nodemap.length
                            && !self.equal.get(self.nodemap.rank0(index + 1) - 1)
                        {
                            min_t_ = max_s_ + max_t_ - min_s_;
                        }
                    }
                    index_t_ = None;
                }

                self._search_window(
                    snapshot,
                    sidelen,
                    top_,
                    bottom_,
                    left_,
                    right_,
                    lower,
                    upper,
                    index_t_,
                    index_s_,
                    min_t_,
                    min_s_,
                    max_t_,
                    max_s_,
                    cells,
                    top_offset_,
                    left_offset_,
                );
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

/// An array of bits.
///
/// This unindexed version is used to build up a bitmap using the `push` method. Once a bitmap is
/// built, it should be converted to an indexed type for performant rank queries.
///
/// Typical usage:
///
/// let mut builder = BitMapBuilder::new();
///
/// builder.push(...)
/// builder.push(...)
/// etc...
///
/// let bitmap = BitMap::from(builder);
///
struct BitMapBuilder {
    length: usize,
    bitmap: Vec<u8>,
}

impl BitMapBuilder {
    /// Initialize an empty BitMapBuilder
    fn new() -> BitMapBuilder {
        BitMapBuilder {
            length: 0,
            bitmap: vec![],
        }
    }

    /// Push a bit onto the BitMapBuilder
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

/// An array of bits with a single level index for making fast rank queries.
///
struct BitMap {
    length: usize,
    k: usize,
    index: Vec<u32>,
    bitmap: Vec<u32>,
}

impl From<BitMapBuilder> for BitMap {
    /// Generate an indexed bitmap from an unindexed one.
    ///
    /// Index is an array of bit counts for every k words in the bitmap, such that
    /// rank(i) = index[i / k / wordlen] if i is an even multiple of k * wordlen. wordlen is 32 for
    /// this implementation which uses 32 bit unsigned integers.
    fn from(bitmap: BitMapBuilder) -> Self {
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

        BitMap {
            length: bitmap.length,
            k,
            index,
            bitmap: bitmap32,
        }
    }
}

impl BitMap {
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
}

/// Compact storage for integers (Directly Addressable Codes)
struct Dacs {
    levels: Vec<(BitMap, Vec<u8>)>,
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

        let n = zigzag_decode(n);
        T::from(n).unwrap()
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
            .map(|(bitmap, bytes)| (BitMap::from(bitmap), bytes))
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
fn rearrange<T>(lower: T, upper: T) -> (T, T)
where
    T: PrimInt + Debug,
{
    if lower > upper {
        (upper, lower)
    } else {
        (lower, upper)
    }
}

#[cfg(test)]
mod tests;
