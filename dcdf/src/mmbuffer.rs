use std::cmp;

use ndarray::{s, ArrayBase, ArrayViewMut1, ArrayViewMut3, Data, Ix3};
use num_traits::{Float, Num};

use crate::{
    fixed::{from_fixed, suggest_fraction, to_fixed, Fraction},
    mmstruct::MMEncoding,
};

pub(crate) enum MMBuffer0 {
    I32(i32),
    I64(i64),
    F32((f32, usize)),
    F64((f64, usize)),
}

impl MMBuffer0 {
    pub fn set(&mut self, value: i64) {
        match self {
            Self::I32(ref mut dest) => {
                *dest = value as i32;
            }
            Self::I64(ref mut dest) => {
                *dest = value;
            }
            Self::F32((ref mut dest, fractional_bits)) => {
                *dest = from_fixed(value, *fractional_bits);
            }
            Self::F64((ref mut dest, fractional_bits)) => {
                *dest = from_fixed(value, *fractional_bits);
            }
        }
    }

    pub fn set_fractional_bits(&mut self, fractional_bits: usize) {
        match self {
            Self::I32(_) | Self::I64(_) => {}
            Self::F32((_, ref mut dest_bits)) => {
                *dest_bits = fractional_bits;
            }
            Self::F64((_, ref mut dest_bits)) => {
                *dest_bits = fractional_bits;
            }
        }
    }
}

impl From<MMBuffer0> for i32 {
    fn from(buffer: MMBuffer0) -> Self {
        match buffer {
            MMBuffer0::I32(value) => value,
            _ => {
                panic!("Not the i32 variant");
            }
        }
    }
}

impl From<MMBuffer0> for i64 {
    fn from(buffer: MMBuffer0) -> Self {
        match buffer {
            MMBuffer0::I64(value) => value,
            _ => {
                panic!("Not the i64 variant");
            }
        }
    }
}

impl From<MMBuffer0> for f32 {
    fn from(buffer: MMBuffer0) -> Self {
        match buffer {
            MMBuffer0::F32((value, _)) => value,
            _ => {
                panic!("Not the f32 variant");
            }
        }
    }
}

impl From<MMBuffer0> for f64 {
    fn from(buffer: MMBuffer0) -> Self {
        match buffer {
            MMBuffer0::F64((value, _)) => value,
            _ => {
                panic!("Not the f64 variant");
            }
        }
    }
}
pub(crate) enum MMBuffer1<'a> {
    I32(MMBuffer1I32<'a>),
    I64(MMBuffer1I64<'a>),
    F32(MMBuffer1F32<'a>),
    F64(MMBuffer1F64<'a>),
}

impl<'a> MMBuffer1<'a> {
    pub(crate) fn set(&mut self, index: usize, value: i64) {
        match self {
            Self::I32(buffer) => buffer.set(index, value),
            Self::I64(buffer) => buffer.set(index, value),
            Self::F32(buffer) => buffer.set(index, value),
            Self::F64(buffer) => buffer.set(index, value),
        }
    }

    pub(crate) fn slice(&mut self, start: usize, end: usize) -> Self {
        match self {
            Self::I32(buffer) => Self::I32(buffer.slice(start, end)),
            Self::I64(buffer) => Self::I64(buffer.slice(start, end)),
            Self::F32(buffer) => Self::F32(buffer.slice(start, end)),
            Self::F64(buffer) => Self::F64(buffer.slice(start, end)),
        }
    }

    pub(crate) fn set_fractional_bits(&mut self, fractional_bits: usize) {
        match self {
            Self::I32(_) | Self::I64(_) => {}
            Self::F32(ref mut buffer) => {
                buffer.fractional_bits = fractional_bits;
            }
            Self::F64(ref mut buffer) => {
                buffer.fractional_bits = fractional_bits;
            }
        }
    }

    pub(crate) fn new_i32(array: ArrayViewMut1<'a, i32>) -> Self {
        Self::I32(MMBuffer1I32(array))
    }

    pub(crate) fn new_i64(array: ArrayViewMut1<'a, i64>) -> Self {
        Self::I64(MMBuffer1I64(array))
    }

    pub(crate) fn new_f32(
        array: ArrayViewMut1<'a, f32>,
        fractional_bits: usize,
        round: bool,
    ) -> Self {
        Self::F32(MMBuffer1F32 {
            array: array,
            fractional_bits: fractional_bits,
            round: round,
        })
    }

    pub(crate) fn new_f64(
        array: ArrayViewMut1<'a, f64>,
        fractional_bits: usize,
        round: bool,
    ) -> Self {
        Self::F64(MMBuffer1F64 {
            array: array,
            fractional_bits: fractional_bits,
            round: round,
        })
    }
}

pub(crate) struct MMBuffer1I64<'a>(ArrayViewMut1<'a, i64>);

impl<'a> MMBuffer1I64<'a> {
    fn set(&mut self, index: usize, value: i64) {
        self.0[[index]] = value;
    }

    fn slice(&mut self, start: usize, end: usize) -> Self {
        let subarray = unsafe {
            self.0
                .slice_mut(s![start..end])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self(subarray)
    }
}

pub(crate) struct MMBuffer1I32<'a>(ArrayViewMut1<'a, i32>);

impl<'a> MMBuffer1I32<'a> {
    fn set(&mut self, index: usize, value: i64) {
        self.0[[index]] = value as i32;
    }

    fn slice(&mut self, start: usize, end: usize) -> Self {
        let subarray = unsafe {
            self.0
                .slice_mut(s![start..end])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self(subarray)
    }
}

pub(crate) struct MMBuffer1F32<'a> {
    array: ArrayViewMut1<'a, f32>,
    fractional_bits: usize,
    round: bool,
}

impl<'a> MMBuffer1F32<'a> {
    fn set(&mut self, index: usize, value: i64) {
        self.array[[index]] = from_fixed(value, self.fractional_bits);
    }

    fn slice(&mut self, start: usize, end: usize) -> Self {
        let subarray = unsafe {
            self.array
                .slice_mut(s![start..end])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self {
            array: subarray,
            fractional_bits: self.fractional_bits,
            round: self.round,
        }
    }
}

pub(crate) struct MMBuffer1F64<'a> {
    array: ArrayViewMut1<'a, f64>,
    fractional_bits: usize,
    round: bool,
}

impl<'a> MMBuffer1F64<'a> {
    fn set(&mut self, index: usize, value: i64) {
        self.array[[index]] = from_fixed(value, self.fractional_bits);
    }

    fn slice(&mut self, start: usize, end: usize) -> Self {
        let subarray = unsafe {
            self.array
                .slice_mut(s![start..end])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self {
            array: subarray,
            fractional_bits: self.fractional_bits,
            round: self.round,
        }
    }
}

pub(crate) enum MMBuffer3<'a> {
    I32(MMBuffer3I32<'a>),
    I64(MMBuffer3I64<'a>),
    F32(MMBuffer3F32<'a>),
    F64(MMBuffer3F64<'a>),
}

impl<'a> MMBuffer3<'a> {
    pub(crate) fn slice(
        &mut self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        match self {
            Self::I32(buffer) => Self::I32(buffer.slice(start, end, top, bottom, left, right)),
            Self::I64(buffer) => Self::I64(buffer.slice(start, end, top, bottom, left, right)),
            Self::F32(buffer) => Self::F32(buffer.slice(start, end, top, bottom, left, right)),
            Self::F64(buffer) => Self::F64(buffer.slice(start, end, top, bottom, left, right)),
        }
    }

    pub(crate) fn set_fractional_bits(&mut self, fractional_bits: usize) {
        match self {
            Self::I32(_) | Self::I64(_) => {}
            Self::F32(ref mut buffer) => {
                buffer.fractional_bits = fractional_bits;
            }
            Self::F64(ref mut buffer) => {
                buffer.fractional_bits = fractional_bits;
            }
        }
    }

    pub(crate) fn set(&mut self, instant: usize, row: usize, col: usize, value: i64) {
        match self {
            Self::I32(buffer) => buffer.set(instant, row, col, value),
            Self::I64(buffer) => buffer.set(instant, row, col, value),
            Self::F32(buffer) => buffer.set(instant, row, col, value),
            Self::F64(buffer) => buffer.set(instant, row, col, value),
        }
    }

    pub(crate) fn get(&self, instant: usize, row: usize, col: usize) -> i64 {
        match self {
            Self::I32(buffer) => buffer.0[[instant, row, col]] as i64,
            Self::I64(buffer) => buffer.0[[instant, row, col]],
            Self::F32(buffer) => buffer.get(instant, row, col),
            Self::F64(buffer) => buffer.get(instant, row, col),
        }
    }

    pub(crate) fn new_i32(array: ArrayViewMut3<'a, i32>) -> Self {
        Self::I32(MMBuffer3I32(array))
    }

    pub(crate) fn new_i64(array: ArrayViewMut3<'a, i64>) -> Self {
        Self::I64(MMBuffer3I64(array))
    }

    pub(crate) fn new_f32(
        array: ArrayViewMut3<'a, f32>,
        fractional_bits: usize,
        round: bool,
    ) -> Self {
        if round && fractional_bits == 0 {}
        Self::F32(MMBuffer3F32 {
            array: array,
            fractional_bits: fractional_bits,
            round: round,
        })
    }

    pub(crate) fn new_f64(
        array: ArrayViewMut3<'a, f64>,
        fractional_bits: usize,
        round: bool,
    ) -> Self {
        Self::F64(MMBuffer3F64 {
            array: array,
            fractional_bits: fractional_bits,
            round: round,
        })
    }

    pub(crate) fn fill_instant(&mut self, i: usize, value: i64) {
        match self {
            Self::I32(buffer) => {
                buffer.0.slice_mut(s![i, .., ..]).fill(value as i32);
            }
            Self::I64(buffer) => {
                buffer.0.slice_mut(s![i, .., ..]).fill(value);
            }
            Self::F32(buffer) => {
                buffer
                    .array
                    .slice_mut(s![i, .., ..])
                    .fill(from_fixed(value, buffer.fractional_bits));
            }
            Self::F64(buffer) => {
                buffer
                    .array
                    .slice_mut(s![i, .., ..])
                    .fill(from_fixed(value, buffer.fractional_bits));
            }
        }
    }

    pub(crate) fn min_max(&self) -> Vec<(i64, i64)> {
        match self {
            Self::I32(buffer) => min_max(&buffer.0)
                .into_iter()
                .map(|(a, b)| (a as i64, b as i64))
                .collect(),

            Self::I64(buffer) => min_max(&buffer.0),

            Self::F32(buffer) => min_max_float(&buffer.array)
                .into_iter()
                .map(|(a, b)| {
                    (
                        to_fixed(a, buffer.fractional_bits, buffer.round),
                        to_fixed(b, buffer.fractional_bits, buffer.round),
                    )
                })
                .collect(),

            Self::F64(buffer) => min_max_float(&buffer.array)
                .into_iter()
                .map(|(a, b)| {
                    (
                        to_fixed(a, buffer.fractional_bits, buffer.round),
                        to_fixed(b, buffer.fractional_bits, buffer.round),
                    )
                })
                .collect(),
        }
    }

    pub(crate) fn fractional_bits(&self) -> usize {
        match self {
            Self::I32(_) | Self::I64(_) => 0,
            Self::F32(buffer) => buffer.fractional_bits,
            Self::F64(buffer) => buffer.fractional_bits,
        }
    }

    pub(crate) fn encoding(&self) -> MMEncoding {
        match self {
            Self::I32(_) => MMEncoding::I32,
            Self::I64(_) => MMEncoding::I64,
            Self::F32(_) => MMEncoding::F32,
            Self::F64(_) => MMEncoding::F64,
        }
    }

    pub(crate) fn shape(&self) -> [usize; 3] {
        let shape = match self {
            Self::I32(buffer) => buffer.0.shape(),
            Self::I64(buffer) => buffer.0.shape(),
            Self::F32(buffer) => buffer.array.shape(),
            Self::F64(buffer) => buffer.array.shape(),
        };

        shape.try_into().unwrap()
    }

    pub(crate) fn compute_fractional_bits(&mut self) {
        match self {
            Self::I32(_) | Self::I64(_) => {}
            Self::F32(buffer) => buffer.compute_fractional_bits(),
            Self::F64(buffer) => buffer.compute_fractional_bits(),
        }
    }
}

fn min_max<N, S>(array: &ArrayBase<S, Ix3>) -> Vec<(N, N)>
where
    N: Num + PartialOrd + Copy,
    S: Data<Elem = N>,
{
    let mut min_max = Vec::with_capacity(array.shape()[0]);
    for subarray in array.outer_iter() {
        let value = subarray[[0, 0]];
        let (min_value, max_value) =
            subarray
                .iter()
                .fold((value, value), |(min_value, max_value), value| {
                    let min_value = if *value < min_value {
                        *value
                    } else {
                        min_value
                    };
                    let max_value = if *value > max_value {
                        *value
                    } else {
                        max_value
                    };
                    (min_value, max_value)
                });

        min_max.push((min_value, max_value));
    }

    min_max
}

fn min_max_float<N, S>(array: &ArrayBase<S, Ix3>) -> Vec<(N, N)>
where
    N: Float + PartialOrd + Copy,
    S: Data<Elem = N>,
{
    let mut min_max = Vec::with_capacity(array.shape()[0]);
    for subarray in array.outer_iter() {
        let mut values = subarray.iter();
        let first_value = values.next().unwrap();
        let (mut min_value, mut max_value) = (first_value, first_value);
        while min_value.is_nan() {
            if let Some(value) = values.next() {
                min_value = value;
                max_value = value;
            } else {
                break;
            }
        }

        for n in values {
            if n.is_nan() {
                min_value = n;
            } else {
                if n < min_value {
                    min_value = n;
                } else if n > max_value {
                    max_value = n;
                }
            }
        }

        min_max.push((*min_value, *max_value));
    }
    min_max
}

pub(crate) struct MMBuffer3I32<'a>(ArrayViewMut3<'a, i32>);

impl<'a> MMBuffer3I32<'a> {
    fn set(&mut self, instant: usize, row: usize, col: usize, value: i64) {
        self.0[[instant, row, col]] = value as i32;
    }

    fn slice(
        &mut self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let subarray = unsafe {
            self.0
                .slice_mut(s![start..end, top..bottom, left..right])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self(subarray)
    }
}
pub(crate) struct MMBuffer3I64<'a>(ArrayViewMut3<'a, i64>);

impl<'a> MMBuffer3I64<'a> {
    fn set(&mut self, instant: usize, row: usize, col: usize, value: i64) {
        self.0[[instant, row, col]] = value;
    }

    fn slice(
        &mut self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let subarray = unsafe {
            self.0
                .slice_mut(s![start..end, top..bottom, left..right])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self(subarray)
    }
}

pub(crate) struct MMBuffer3F32<'a> {
    array: ArrayViewMut3<'a, f32>,
    pub fractional_bits: usize,
    round: bool,
}

impl<'a> MMBuffer3F32<'a> {
    fn set(&mut self, instant: usize, row: usize, col: usize, value: i64) {
        self.array[[instant, row, col]] = from_fixed(value, self.fractional_bits);
    }

    fn get(&self, instant: usize, row: usize, col: usize) -> i64 {
        to_fixed(
            self.array[[instant, row, col]],
            self.fractional_bits,
            self.round,
        )
    }

    fn slice(
        &mut self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let subarray = unsafe {
            self.array
                .slice_mut(s![start..end, top..bottom, left..right])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self {
            array: subarray,
            fractional_bits: self.fractional_bits,
            round: self.round,
        }
    }

    fn compute_fractional_bits(&mut self) {
        let suggestion = suggest_fraction(self.array.view());
        let (round, computed_bits) = match suggestion {
            Fraction::Precise(computed_bits) => (false, computed_bits),
            Fraction::Round(computed_bits) => (true, computed_bits),
        };
        let computed_bits = if self.round {
            cmp::min(computed_bits, self.fractional_bits)
        } else {
            if round {
                panic!("loss of precision");
            } else {
                computed_bits
            }
        };

        self.fractional_bits = computed_bits;
    }
}

pub(crate) struct MMBuffer3F64<'a> {
    array: ArrayViewMut3<'a, f64>,
    pub fractional_bits: usize,
    round: bool,
}

impl<'a> MMBuffer3F64<'a> {
    fn set(&mut self, instant: usize, row: usize, col: usize, value: i64) {
        self.array[[instant, row, col]] = from_fixed(value, self.fractional_bits);
    }

    fn get(&self, instant: usize, row: usize, col: usize) -> i64 {
        to_fixed(
            self.array[[instant, row, col]],
            self.fractional_bits,
            self.round,
        )
    }

    fn slice(
        &mut self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let subarray = unsafe {
            self.array
                .slice_mut(s![start..end, top..bottom, left..right])
                .raw_view_mut()
                .deref_into_view_mut()
        };

        Self {
            array: subarray,
            fractional_bits: self.fractional_bits,
            round: self.round,
        }
    }

    fn compute_fractional_bits(&mut self) {
        let suggestion = suggest_fraction(self.array.view());
        let (round, computed_bits) = match suggestion {
            Fraction::Precise(computed_bits) => (false, computed_bits),
            Fraction::Round(computed_bits) => (true, computed_bits),
        };
        let computed_bits = if self.round {
            cmp::min(computed_bits, self.fractional_bits)
        } else {
            if round {
                panic!("loss of precision");
            } else {
                computed_bits
            }
        };

        self.fractional_bits = computed_bits;
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed::to_fixed;

    use super::*;

    mod mmbuffer1 {
        use ndarray::Array1;

        use super::*;

        #[test]
        fn set_i32() {
            let mut a = Array1::zeros([8]);
            let mut buf = MMBuffer1::new_i32(a.view_mut());
            buf.set(1, 4);

            assert_eq!(a[[1]], 4);
        }

        #[test]
        fn set_i64() {
            let mut a = Array1::zeros([8]);
            let mut buf = MMBuffer1::new_i64(a.view_mut());
            buf.set(1, 4);

            assert_eq!(a[[1]], 4);
        }

        #[test]
        fn set_f32() {
            let mut a = Array1::zeros([8]);
            let mut buf = MMBuffer1::new_f32(a.view_mut(), 3, false);
            buf.set(1, to_fixed(1.625, 3, false));

            assert_eq!(a[[1]], 1.625);
        }

        #[test]
        fn set_f64() {
            let mut a = Array1::zeros([8]);
            let mut buf = MMBuffer1::new_f64(a.view_mut(), 3, false);
            buf.set(1, to_fixed(1.625_f64, 3, false));

            assert_eq!(a[[1]], 1.625);
        }
    }

    mod mmbuffer3 {
        use ndarray::Array3;

        use super::*;

        #[test]
        fn set_i32() {
            let mut a = Array3::zeros([8, 8, 8]);
            let mut buf = MMBuffer3::new_i32(a.view_mut());
            buf.set(1, 2, 3, 4);

            assert_eq!(a[[1, 2, 3]], 4);
        }

        #[test]
        fn set_i64() {
            let mut a = Array3::zeros([8, 8, 8]);
            let mut buf = MMBuffer3::new_i64(a.view_mut());
            buf.set(1, 2, 3, 4);

            assert_eq!(a[[1, 2, 3]], 4);
        }

        #[test]
        fn set_f32() {
            let mut a = Array3::zeros([8, 8, 8]);
            let mut buf = MMBuffer3::new_f32(a.view_mut(), 3, false);
            buf.set(1, 2, 3, to_fixed(1.625, 3, false));

            assert_eq!(a[[1, 2, 3]], 1.625);
        }

        #[test]
        fn set_f64() {
            let mut a = Array3::zeros([8, 8, 8]);
            let mut buf = MMBuffer3::new_f64(a.view_mut(), 3, false);
            buf.set(1, 2, 3, to_fixed(1.625_f64, 3, false));

            assert_eq!(a[[1, 2, 3]], 1.625);
        }

        #[test]
        fn get_i32() {
            let mut a = Array3::zeros([8, 8, 8]);
            a[[1, 2, 3]] = 42;
            let buf = MMBuffer3::new_i32(a.view_mut());
            assert_eq!(buf.get(1, 2, 3), 42);
        }

        #[test]
        fn get_i64() {
            let mut a = Array3::zeros([8, 8, 8]);
            a[[1, 2, 3]] = 42;
            let buf = MMBuffer3::new_i64(a.view_mut());
            assert_eq!(buf.get(1, 2, 3), 42);
        }

        #[test]
        fn get_f32() {
            let mut a = Array3::zeros([8, 8, 8]);
            a[[1, 2, 3]] = 1.625;
            let buf = MMBuffer3::new_f32(a.view_mut(), 3, false);
            assert_eq!(buf.get(1, 2, 3), to_fixed(1.625, 3, false));
        }

        #[test]
        fn get_f64() {
            let mut a = Array3::zeros([8, 8, 8]);
            a[[1, 2, 3]] = 1.625;
            let buf = MMBuffer3::new_f64(a.view_mut(), 3, false);
            assert_eq!(buf.get(1, 2, 3), to_fixed(1.625, 3, false));
        }
    }
}
