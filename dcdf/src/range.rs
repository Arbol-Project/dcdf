use std::fmt::Debug;

use ndarray::Array1;
use num_traits::{cast, Float, PrimInt};

#[derive(Clone)]
pub struct FloatRange<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub start: N,
    pub step: N,
    pub steps: usize,
}

impl<N> FloatRange<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn new(start: N, step: N, steps: usize) -> Self {
        Self { start, step, steps }
    }

    pub fn get(&self, index: usize) -> N {
        self.check_bounds(index);
        N::from(index).unwrap() * self.step + self.start
    }

    pub fn slice(&self, start: usize, stop: usize) -> Array1<N> {
        self.check_bounds(stop - 1);
        let start = N::from(start).unwrap() * self.step + self.start;
        let stop = N::from(stop).unwrap() * self.step + self.start;

        Array1::range(start, stop, self.step)
    }

    pub fn len(&self) -> usize {
        self.steps
    }

    pub fn shape(&self) -> [usize; 1] {
        [self.steps]
    }

    pub fn check_bounds(&self, index: usize) {
        if index >= self.steps {
            panic!(
                "Out of bounds: index {index} is out of bounds for array with length {}",
                self.steps
            );
        }
    }
}

#[derive(Clone)]
pub struct IntRange<N>
where
    N: PrimInt + Debug + Send + Sync + 'static,
{
    pub start: N,
    pub step: N,
    pub steps: usize,
}

impl<N> IntRange<N>
where
    N: PrimInt + Debug + Send + Sync + 'static,
{
    pub fn new(start: N, step: N, steps: usize) -> Self {
        Self { start, step, steps }
    }

    pub fn get(&self, index: usize) -> N {
        self.check_bounds(index);
        N::from(index).unwrap() * self.step + self.start
    }

    pub fn slice(&self, start: usize, stop: usize) -> Array1<N> {
        self.check_bounds(stop - 1);
        let start = N::from(start).unwrap() * self.step + self.start;
        let stop = N::from(stop).unwrap() * self.step + self.start;

        // We would prefer to just use:
        //
        //   Array1::from_iter((start..stop).step_by(self.step))
        //
        // Unfortunately there is an issue where you can't make ranges using num_traits::PrimInt so
        // I've worked around it by implementing Iterator on IntRange. There is a decent chance
        // future versions of Rust and/or the num-traits crate will render this a non-issue and we
        // can get rid of the Iterator implementation for IntRange.
        //
        let range = Self::new(start, self.step, cast((stop - start) / self.step).unwrap());

        Array1::from_iter(range)
    }

    pub fn len(&self) -> usize {
        self.steps
    }

    pub fn shape(&self) -> [usize; 1] {
        [self.steps]
    }

    pub fn check_bounds(&self, index: usize) {
        if index >= self.steps {
            panic!(
                "Out of bounds: index {index} is out of bounds for array with length {}",
                self.steps
            );
        }
    }
}

/// See note in IntRange::slice for why this is necessary. Hopefully it can be removed some day.
///
impl<N> Iterator for IntRange<N>
where
    N: PrimInt + Debug + Send + Sync + 'static,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        if self.steps > 0 {
            let next = self.start;
            self.start = self.start + self.step;
            self.steps -= 1;

            Some(next)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{array, s};
    use paste::paste;

    macro_rules! int_range_tests {
        ($name:ident) => {
            paste! {
                #[test]
                fn [<$name _test_get>]() {
                    let (data, range) = $name();
                    for i in 0..range.len() {
                        assert_eq!(range.get(i), data[i]);
                    }
                }

                #[test]
                #[should_panic]
                fn [<$name _test_get_out_bounds>]() {
                    let (_data, range) = $name();
                    assert_eq!(range.get(range.len()), 130); // Out of bounds
                }

                #[test]
                fn [<$name _test_slice>]() {
                    let (data, range) = $name();
                    for i in 0..range.len() / 2 {
                        let (start, end) = (i, range.len() - i);
                        assert_eq!(range.slice(start, end), data.slice(s![start..end]));
                    }
                }

                #[test]
                #[should_panic]
                fn [<$name _test_slice_out_of_bounds>]() {
                    let (_data, range) = $name();
                    let start = range.len() - 1;
                    let end = start + 2;
                    assert_eq!(range.slice(start, end), array![125, 130]);
                }
            }
        };
    }

    macro_rules! float_range_tests {
        ($name:ident) => {
            paste! {
                #[test]
                fn [<$name _test_get>]() {
                    let (data, range) = $name();
                    for i in 0..range.len() {
                        assert_eq!(range.get(i), data[i]);
                    }
                }

                #[test]
                #[should_panic]
                fn [<$name _test_get_out_bounds>]() {
                    let (_data, range) = $name();
                    assert_eq!(range.get(range.len()), 130.0); // Out of bounds
                }

                #[test]
                fn [<$name _test_slice>]() {
                    let (data, range) = $name();
                    for i in 0..range.len() / 2 {
                        let (start, end) = (i, range.len() - i);
                        assert_eq!(range.slice(start, end), data.slice(s![start..end]));
                    }
                }

                #[test]
                #[should_panic]
                fn [<$name _test_slice_out_of_bounds>]() {
                    let (_data, range) = $name();
                    let start = range.len() - 1;
                    let end = start + 2;
                    assert_eq!(range.slice(start, end), array![125.0, 130.0]);
                }
            }
        };
    }

    fn range_i32() -> (Array1<i32>, IntRange<i32>) {
        let data = Array1::from_iter((-20..130).step_by(5));
        let range = IntRange::new(-20, 5, 30);

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        (data, range)
    }

    int_range_tests!(range_i32);

    fn range_i64() -> (Array1<i64>, IntRange<i64>) {
        let data = Array1::from_iter((-20..130).step_by(5));
        let range = IntRange::new(-20, 5, 30);

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        (data, range)
    }

    int_range_tests!(range_i64);

    fn range_f32() -> (Array1<f32>, FloatRange<f32>) {
        let data = Array1::range(-20.0, 130.0, 5.0);
        let range = FloatRange::new(-20.0, 5.0, 30);

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        (data, range)
    }

    float_range_tests!(range_f32);

    fn range_f64() -> (Array1<f64>, FloatRange<f64>) {
        let data = Array1::range(-20.0, 130.0, 5.0);
        let range = FloatRange::new(-20.0, 5.0, 30);

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        (data, range)
    }

    float_range_tests!(range_f64);
}
