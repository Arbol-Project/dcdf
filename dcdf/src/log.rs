use std::{cmp::min, collections::VecDeque};

use async_trait::async_trait;
use futures::io::{AsyncRead, AsyncWrite};

use crate::{
    bitmap::{BitMap, BitMapBuilder},
    cache::Cacheable,
    dac::Dac,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
    snapshot::Snapshot,
};

/// K²-Raster encoded Log
///
/// A Log stores raster data for a particular time instant in a raster time series as the
/// difference between this time instant and a reference Snapshot.
///
pub(crate) struct Log {
    /// Bitmap of tree structure, known as T in Silva-Coira
    nodemap: BitMap,

    /// Bitmap of tree nodes that match referenced snapshot, or have cells that all differ by the
    /// same amount, known as eqB in Silva-Coira paper
    equal: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    max: Dac,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    min: Dac,

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

#[async_trait]
impl Serialize for Log {
    /// Write a log to a stream
    ///
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte(self.k as u8).await?;
        stream.write_u32(self.shape[0] as u32).await?;
        stream.write_u32(self.shape[1] as u32).await?;
        stream.write_u32(self.sidelen as u32).await?;
        self.nodemap.write_to(stream).await?;
        self.equal.write_to(stream).await?;
        self.max.write_to(stream).await?;
        self.min.write_to(stream).await?;

        Ok(())
    }

    /// Read a log from a stream
    ///
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let k = stream.read_byte().await? as i32;
        let shape = [
            stream.read_u32().await? as usize,
            stream.read_u32().await? as usize,
        ];
        let sidelen = stream.read_u32().await? as usize;
        let nodemap = BitMap::read_from(stream).await?;
        let equal = BitMap::read_from(stream).await?;
        let max = Dac::read_from(stream).await?;
        let min = Dac::read_from(stream).await?;

        Ok(Self {
            nodemap,
            equal,
            max,
            min,
            k,
            shape,
            sidelen,
        })
    }
}

impl Cacheable for Log {
    /// Return number of bytes in serialized representation
    ///
    fn size(&self) -> u64 {
        1 + 4 + 4 + 4 + self.nodemap.size() + self.equal.size() + self.max.size() + self.min.size()
    }
}

impl Log {
    /// Build a log from a pair of two-dimensional arrays.
    ///
    /// The notional two-dimensional arrays are represented by `get_s' and `get_t`, which are
    /// functions that take a row and column as arguments and return an i64. The dimensions of the
    /// two-dimensional arrays are given by `shape`. `k` is the K from K²-Raster. The recommended
    /// value is 2. See the literature.
    ///
    /// See the documentation for `Snapshot::build` for the rationale behind the `get_s` and
    /// `get_t` indirection.
    ///

    pub(crate) fn build<GS, GT>(get_s: GS, get_t: GT, shape: [usize; 2], k: i32) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        let mut nodemap = BitMapBuilder::new();
        let mut equal = BitMapBuilder::new();
        let mut max: Vec<i64> = vec![];
        let mut min: Vec<i64> = vec![];

        // Compute the smallest square with sides whose length is a power of K that will contain
        // the passed in data.
        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let root = K2PTreeNode::build(get_s, get_t, shape, k, sidelen);
        let mut to_traverse = VecDeque::new();
        to_traverse.push_back(&root);

        // Breadth first traversal
        while let Some(node) = to_traverse.pop_front() {
            max.push(node.max_t.unwrap_or(0) - node.max_s.unwrap_or(0));

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
                    min.push(node.min_t.unwrap() - node.min_s.unwrap());
                    for child in &node.children {
                        to_traverse.push_back(child);
                    }
                }
            }
        }

        Log {
            nodemap: nodemap.finish(),
            equal: equal.finish(),
            max: Dac::from(max),
            min: Dac::from(min),
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
    pub(crate) fn get(&self, snapshot: &Snapshot, row: usize, col: usize) -> i64 {
        let max_t = self.max.get(0);
        let max_s = snapshot.max.get(0);
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

    fn _get(
        &self,
        snapshot: &Snapshot,
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
                max_s = max_s - snapshot.max.get(index);
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
    /// The passed in `set` function  inserts into a notional two dimensional array that has been
    /// preallocated with the correct dimensions. This indirection allows higher layers to
    /// preallocate a 3 dimensional array at the beginning of a time series query, and provides a
    /// means of injecting data conversion from the underlying stored data to the desired output
    /// numeric type.
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub(crate) fn fill_window<S>(&self, mut set: S, snapshot: &Snapshot, bounds: &geom::Rect)
    where
        S: FnMut(usize, usize, i64),
    {
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        if single_t && (single_s || !self.equal.get(0)) {
            // Both trees have single node or log has single node but it contains a uniform value
            // for all cells
            let max_t: i64 = self.max.get(0);
            let max_s: i64 = snapshot.max.get(0);
            for row in 0..bounds.rows() {
                for col in 0..bounds.cols() {
                    set(row, col, max_t + max_s);
                }
            }
        } else {
            self._fill_window(
                &mut set,
                snapshot,
                self.sidelen,
                bounds.top,
                bounds.bottom - 1,
                bounds.left,
                bounds.right - 1,
                if single_t { None } else { Some(0) },
                if single_s { None } else { Some(0) },
                self.max.get(0),
                snapshot.max.get(0),
                bounds.top,
                bounds.left,
                0,
                0,
            );
        }
    }

    fn _fill_window<S>(
        &self,
        set: &mut S,
        snapshot: &Snapshot,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: i64,
        max_s: i64,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
    ) where
        S: FnMut(usize, usize, i64),
    {
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
                    Some(index) => max_s - snapshot.max.get(index),
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
                            set(
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                                value,
                            );
                        }
                    }
                } else if leaf_s {
                    self._fill_window(
                        set,
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
                                        set(
                                            top_offset_ + row - window_top,
                                            left_offset_ + col - window_left,
                                            value,
                                        );
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    self._fill_window(
                        set,
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
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                    );
                } else {
                    self._fill_window(
                        set,
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
    pub(crate) fn search_window(
        &self,
        snapshot: &Snapshot,
        bounds: &geom::Rect,
        lower: i64,
        upper: i64,
    ) -> Vec<(usize, usize)> {
        let mut cells: Vec<(usize, usize)> = vec![];
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        self._search_window(
            snapshot,
            self.sidelen,
            bounds.top,
            bounds.bottom - 1,
            bounds.left,
            bounds.right - 1,
            lower,
            upper,
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
        snapshot: &Snapshot,
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
                    Some(index) => max_s - snapshot.max.get(index),
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
                            min_s + snapshot.min.get(snapshot.nodemap.rank(index))
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
}

// Temporary tree structure for building T - K^2 raster
struct K2PTreeNode {
    max_t: Option<i64>,
    min_t: Option<i64>,
    max_s: Option<i64>,
    min_s: Option<i64>,
    diff: i64,
    equal: bool,
    children: Vec<K2PTreeNode>,
}

impl K2PTreeNode {
    fn build<GS, GT>(get_s: GS, get_t: GT, shape: [usize; 2], k: i32, sidelen: usize) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        Self::_build(&get_s, &get_t, shape, k as usize, sidelen, 0, 0)
    }

    fn _build<GS, GT>(
        get_s: &GS,
        get_t: &GT,
        shape: [usize; 2],
        k: usize,
        sidelen: usize,
        row: usize,
        col: usize,
    ) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        // Leaf node
        if sidelen == 1 {
            let [rows, cols] = shape;
            let value_s = if row < rows && col < cols {
                Some(get_s(row, col))
            } else {
                None
            };
            let value_t = if row < rows && col < cols {
                Some(get_t(row, col))
            } else {
                None
            };
            let diff = value_t.unwrap_or(0) - value_s.unwrap_or(0);
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
        let mut children: Vec<K2PTreeNode> = vec![];
        let sidelen = sidelen / k;
        for i in 0..k {
            let row_ = row + i * sidelen;
            for j in 0..k {
                let col_ = col + j * sidelen;
                children.push(K2PTreeNode::_build(
                    get_s, get_t, shape, k, sidelen, row_, col_,
                ));
            }
        }

        let mut max_t = children[0].max_t;
        let mut min_t = children[0].min_t;
        let mut max_s = children[0].max_s;
        let mut min_s = children[0].min_s;
        let mut equal = children.iter().all(|child| child.equal);
        let diff = children[0].diff;

        fn is_lt(left: Option<i64>, right: Option<i64>) -> bool {
            if let Some(left) = left {
                if let Some(right) = right {
                    return left < right;
                }
            }
            false
        }

        for child in &children[1..] {
            if is_lt(max_t, child.max_t) {
                max_t = child.max_t;
            }
            if is_lt(child.min_t, min_t) {
                min_t = child.min_t;
            }
            if is_lt(max_s, child.max_s) {
                max_s = child.max_s;
            }
            if is_lt(child.min_s, min_s) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::array_search_window2;
    use futures::io::Cursor;
    use ndarray::{arr3, s, Array2, Array3};
    use std::collections::HashSet;

    fn array8() -> Array3<i64> {
        arr3(&[
            [
                [9, 8, 7, 7, 6, 6, 3, 2],
                [7, 7, 7, 7, 6, 6, 3, 3],
                [6, 6, 6, 6, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 3, 5, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 7, 7, 2, 2],
                [7, 7, 7, 7, 7, 7, 2, 2],
                [6, 6, 6, 6, 4, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 5, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 8, 7, 5, 5],
                [7, 7, 7, 7, 7, 7, 5, 5],
                [7, 7, 6, 6, 4, 3, 4, 4],
                [6, 6, 6, 6, 4, 4, 4, 4],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 6, 4, 4, 4],
                [4, 4, 4, 4, 5, 4, 4, 4],
            ],
        ])
    }

    fn array9() -> Array3<i64> {
        arr3(&[
            [
                [9, 8, 7, 7, 6, 6, 3, 2, 1],
                [7, 7, 7, 7, 6, 6, 3, 3, 3],
                [6, 6, 6, 6, 3, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 3, 5, 4, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 7, 7, 2, 2, 2],
                [7, 7, 7, 7, 7, 7, 2, 2, 2],
                [6, 6, 6, 6, 4, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 4, 5, 5, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 5],
                [4, 4, 4, 4, 5, 4, 4, 4, 1],
            ],
            [
                [9, 8, 7, 7, 8, 7, 5, 5, 2],
                [7, 7, 7, 7, 7, 7, 5, 5, 2],
                [7, 7, 6, 6, 4, 3, 4, 4, 3],
                [6, 6, 6, 6, 4, 4, 4, 4, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 4, 5, 6, 4, 4, 4, 4],
                [4, 4, 4, 4, 5, 4, 4, 4, 4],
                [5, 4, 4, 4, 5, 5, 5, 5, 10],
            ],
        ])
    }

    #[test]
    fn build() {
        let data = array8();
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        assert_eq!(log.nodemap.length, 17);
        assert_eq!(log.nodemap.bitmap, vec![0b10111001000010010000000000000000]);
        assert_eq!(log.equal.length, 10);
        assert_eq!(log.equal.bitmap, vec![0b10001010000000000000000000000000]);

        assert_eq!(
            log.max.collect(),
            vec![
                0, 0, 1, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                0, 0
            ]
        );

        assert_eq!(log.min.collect(), vec![0, 0, 0, 0, 0, 1, 0,]);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        assert_eq!(log.nodemap.length, 21);
        assert_eq!(log.nodemap.bitmap, vec![0b11111000010100001001000000000000]);
        assert_eq!(log.equal.length, 12);
        assert_eq!(log.equal.bitmap, vec![0b10100010100000000000000000000000]);

        assert_eq!(
            log.max.collect(),
            vec![
                0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 1, 1, 1, 0, 1,
                1, 1, 0, 1, 0, 2, 0, 1, 0,
            ]
        );

        assert_eq!(log.min.collect(), vec![1, 1, 1, 0, 0, 1, 0, 1, 0,]);

        assert_eq!(log.shape, [8, 8]);
    }

    #[test]
    fn build_make_sure_fill_values_match_local_nonfill_values_in_same_quadbox() {
        let mut data: Array3<i64> = Array3::zeros([3, 9, 9]) + 5;
        data.slice_mut(s![.., ..8, ..8]).assign(&array8());
        data.slice_mut(s![0, .., ..])
            .assign(&array9().slice(s![0, .., ..]));

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);

        // Copmared to the previous test with an 8x8 array, expecting only 4 new nodes because fill
        // values for expanded 16x16 array will all be 5 because quadboxes except for upper left
        // all contain only 5s or the fill value.
        assert_eq!(log.nodemap.length, 21);

        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);
        assert_eq!(log.get(&snapshot, 8, 8), 5);
    }

    #[test]
    fn get() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_single_node_trees() {
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), 42);
            }
        }
    }

    #[test]
    fn get_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 20;
        let data_t = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data_t[[row, col]]);
            }
        }
    }

    #[test]
    fn get_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 20;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 1..8 {
                assert_eq!(log.get(&snapshot, row, col), data_t[[row, col]]);
            }
        }
    }

    #[test]
    fn get_equal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);

        for row in 0..8 {
            for col in 1..8 {
                assert_eq!(log.get(&snapshot, row, col), data_s[[row, col]]);
            }
        }
    }

    #[test]
    fn get_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 3);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 3);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 3);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 3);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_trees() {
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 42;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 20;
        let data_t = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 20;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_equal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = log.get_window(&snapshot, &bounds);
                        let expected = data_s.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9() {
        let data = array9();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9_k3() {
        let data = array9();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 3);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_node_trees() {
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i64> = Array2::zeros([8, 8]) + 42;
        let data_t = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i64> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_sequal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window2(
                                    data_s.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let bounds = geom::Rect::new(top, bottom, left, right);
                                let coords = log.search_window(&snapshot, &bounds, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_no_values_in_range() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let coords = log.search_window(&snapshot, &bounds, 100, 200);
                        assert_eq!(coords.len(), 0);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn serialize_deserialize() -> Result<()> {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);

        let mut buffer: Vec<u8> = Vec::with_capacity(log.size() as usize);
        log.write_to(&mut buffer).await?;
        assert_eq!(buffer.len(), log.size() as usize);

        let mut buffer = Cursor::new(buffer);
        let log = Log::read_from(&mut buffer).await?;

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        Ok(())
    }
}
