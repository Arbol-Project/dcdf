//! Functions related to converting floating point numbers to an internal fixed point
//! representation and back again.
//!
//! This is for internal use by DCDF. Number of fractional bits is not stored in the fixed point
//! representation. Instead, that information is stored once for a Chunk, where it is assumed all
//! numbers stored in that Chunk are encoded with the same number of fractional bits.
//!
//! In order to be able to encode NaN, which is encoded as 0, finite numbers will always be encoded
//! with the least significant bit set to 1. This bit is elided when decoding and is not used as
//! part of the numerical value.

use ndarray::Array2;
use num_traits::Float;
use std::fmt::Debug;

/// Convert floating point number to fixed point format
///
/// # Arguments
///
/// * `n` - A floating point number to convert to fixed point
/// * `fractional_bits` - The number of bits to use for the fractional part.
/// * `round` - Whether values should be rounded or not. If `false`, loss of numeric precision will
///     cause a panic. Otherwise, values are rounded to fit in the number of fractional bits.
///
/// # Panics
///
/// * When `round` is `false` and  a loss of precision would occur.
/// * When fixed point representation overflows as i64 (ie the number is too big)
/// * When `n` is inf, or -inf.
///
pub fn to_fixed<F>(n: F, fractional_bits: usize, round: bool) -> i64
where
    F: Float + Debug,
{
    if n.is_nan() {
        return 0;
    }

    if !n.is_finite() {
        panic!("Cannot convert {n:?} to fixed point representation.");
    }

    // Shift the point fractional_bits bits to the left
    let mut shifted = n * F::from(1 << fractional_bits).unwrap();

    // Check to make sure we don't have any bits to the right of the point after shifting
    if shifted.fract() > F::zero() {
        if round {
            shifted = shifted.round();
        } else {
            panic!(
                "\
                Converting {:?} to fixed point representation with {} fractional bits \
                results in loss of precision. For lossy conversion you can pass `true` for \
                `round`.",
                n, fractional_bits
            );
        }
    }

    // Shift left one more bit so that LSB can be used to indicate NaN
    shifted = shifted * F::from(2).unwrap();

    match shifted.to_i64() {
        Some(number) => number + 1, // Set LSB to 1 to indicate not a NaN
        None => panic!("Overflow converting {n:?} to fixed point representation."),
    }
}

/// Convert from fixed point representation back to a floating point number
///
/// # Arguments
///
/// * `n` - The fixed point number to convert to floating point
/// * `fractional_bits` - The number of bits used for the fractional part in the fixed point
///     representation.
///
pub fn from_fixed<F: Float>(n: i64, fractional_bits: usize) -> F {
    match n {
        0 => F::nan(),
        _ => F::from(n - 1).unwrap() / F::from(1 << (fractional_bits + 1)).unwrap(),
    }
}

#[derive(Copy, Clone)]
pub enum Fraction {
    Precise(usize),
    Round(usize),
}

pub use Fraction::{Precise, Round};

pub enum Continue {
    Yes,
    No,
}

pub struct FractionSuggester<F>
where
    F: Float + Debug,
{
    max_value: F,
    max_fraction_bits: usize,
    fraction_bits: usize,
    round: bool,
}

impl<F> FractionSuggester<F>
where
    F: Float + Debug,
{
    // Basic gist is figure out how many bits we need for the whole number part, using the
    // max_value passed in. From there shift each value in the dataset as far to the left as
    // possible given the number of whole number bits needed, then look at the number of trailing
    // zeros on each shifted value to determine how many actual fractional bits we need for that
    // number, and return the maximum number.

    pub fn new(max_value: F) -> Self {
        // DACs can store 64 bits. We lose 1 bit to the sign for signed numbers, and we lose
        // another bit to the encoding used by "fixed" to differentiate finite numbers from NaN.
        const TOTAL_BITS: usize = 62;
        let whole_bits = 1 + max_value.to_f64().unwrap().log2().floor() as usize;

        Self {
            max_value,
            max_fraction_bits: TOTAL_BITS - whole_bits,
            fraction_bits: 0,
            round: false,
        }
    }

    pub fn push(&mut self, instant: Array2<F>) -> Continue {
        if self.round {
            panic!("FractionSuggester.push() called again after returning Continue::No");
        }

        for n in instant {
            let n: f64 = n.to_f64().unwrap();
            if n.is_nan() {
                continue;
            }
            let shifted = n * (1_i64 << self.max_fraction_bits) as f64;

            // If we've left shifted a number as far as it will go and we still have a fractional
            // part, then this dataset will need to be rounded and there will be some loss of
            // precision.
            if shifted.fract() != 0.0 {
                self.fraction_bits = self.max_fraction_bits;
                self.round = true;
                return Continue::No;
            }

            let shifted = shifted as i64;
            if shifted == i64::MAX {
                // Conversion from float to int saturates on overflow, so assume there was an
                // overflow if result is MAX
                panic!("Value {n} is greater than max_value {:?}", self.max_value);
            }

            let these_bits = self
                .max_fraction_bits
                .saturating_sub(shifted.trailing_zeros() as usize);
            if these_bits > self.fraction_bits {
                self.fraction_bits = these_bits;
            }
        }

        Continue::Yes
    }

    pub fn finish(self) -> Fraction {
        if self.round {
            Round(self.fraction_bits)
        } else {
            Precise(self.fraction_bits)
        }
    }
}

pub fn suggest_fraction<F, T>(instants: T, max_value: F) -> Fraction
where
    F: Float + Debug,
    T: Iterator<Item = Array2<F>>,
{
    let mut suggester = FractionSuggester::new(max_value);

    for instant in instants {
        match suggester.push(instant) {
            Continue::No => break,
            Continue::Yes => continue,
        }
    }

    suggester.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn array() -> Vec<Array2<f32>> {
        let data = vec![
            arr2(&[
                [9.5, 8.25, 7.75, 7.75, 6.125, 6.125, 3.375, 2.625],
                [7.75, 7.75, 7.75, 7.75, 6.125, 6.125, 3.375, 3.375],
                [6.125, 6.125, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 3.375, 5.0, 4.875, 4.875, 4.875, 4.875],
                [4.875, 4.875, 3.375, 4.875, 4.875, 4.875, 4.875, 4.875],
            ]),
            arr2(&[
                [9.5, 8.25, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
                [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
                [6.125, 6.125, 6.125, 6.125, 4.875, 3.375, 3.375, 3.375],
                [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 4.875, 5.0, 5.0, 4.875, 4.875, 4.875],
                [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
            ]),
            arr2(&[
                [9.5, 8.25, 7.75, 7.75, 8.25, 7.75, 5.0, 5.0],
                [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 5.0, 5.0],
                [7.75, 7.75, 6.125, 6.125, 4.875, 3.375, 4.875, 4.875],
                [6.125, 6.125, 6.125, 6.125, 4.875, 4.875, 4.875, 4.875],
                [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                [3.375, 3.375, 4.875, 5.0, 6.125, 4.875, 4.875, 4.875],
                [4.875, 4.875, 4.875, 4.875, 5.0, 4.875, 4.875, 4.875],
            ]),
        ];

        data.into_iter().cycle().take(100).collect()
    }

    #[test]
    fn test_to_fixed() {
        let n = 1.5; // 0b1.1
        assert_eq!(to_fixed(n, 1, false), 7); // 0b111
        assert_eq!(to_fixed(-n, 1, false), -5);
        assert_eq!(to_fixed(n, 8, false), 769); // 0b1100000001

        let n = 0.0625; // 0b0.0001
        assert_eq!(to_fixed(n, 4, false), 3); // 0b11

        let n = 0.0;
        assert_eq!(to_fixed(n, 16, false), 1); // 0b1

        let n = -0.0;
        assert_eq!(to_fixed(n, 16, false), 1); // 0b1
    }

    #[test]
    fn test_to_fixed_round() {
        let n = 1.5;
        assert_eq!(to_fixed(n, 1, true), 7);
        assert_eq!(to_fixed(n, 8, true), 769);

        let n = 0.0625;
        assert_eq!(to_fixed(n, 4, true), 3);
        assert_eq!(to_fixed(n, 3, true), 3);
        assert_eq!(to_fixed(n, 2, true), 1);

        // 1/10 cannot be precisely represented in binary
        let n = 0.1;
        assert_eq!(to_fixed(n, 16, true), 6554 * 2 + 1);

        let n = 0.0;
        assert_eq!(to_fixed(n, 16, true), 1);

        let n = -0.0;
        assert_eq!(to_fixed(n, 16, true), 1);
    }

    #[test]
    fn test_from_fixed() {
        assert_eq!(from_fixed::<f32>(7, 1), 1.5);
        assert_eq!(from_fixed::<f32>(-5, 1), -1.5);
        assert_eq!(from_fixed::<f64>(769, 8), 1.5);

        assert_eq!(from_fixed::<f32>(3, 4), 0.0625);
        assert_eq!(from_fixed::<f32>(1, 13), 0.0);

        // 1/10 cannot be precisely represented in binary
        assert!(from_fixed::<f64>(6554 * 2 + 1, 16) - 0.1 < 1e-5);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_loss_of_precision_off_by_1() {
        let n = 0.0625; // 0b0.0001
        to_fixed(n, 3, false);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_loss_of_precision_off_by_more() {
        let n = 1.0625; // 0b1.0001
        to_fixed(n, 3, false);
    }

    #[test]
    fn test_to_fixed_nan() {
        let n = std::f64::NAN;
        assert_eq!(to_fixed(n, 12, false), 0);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_inf() {
        let n = std::f64::INFINITY;
        to_fixed(n, 14, false);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_neg_inf() {
        let n = std::f64::NEG_INFINITY;
        to_fixed(n, 14, false);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_overflow() {
        let n = 1.5e100;
        to_fixed(n, 1, false);
    }

    #[test]
    fn suggest_fraction_3bits() {
        let data = array();
        let fraction = suggest_fraction(data.into_iter(), 15.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 3);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_4bits() {
        let data = vec![arr2(&[[16.0, 1.0 / 16.0]])];
        let fraction = suggest_fraction(data.into_iter(), 16.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 4);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_all_the_bits() {
        let data = vec![
            // 1/10 is infinite repeating digits in binary, like 1/3 in decimal. We still don't end
            // up rounding, though, because we can represent all the bits that are in the f64
            // representation. This doesn't introduce any more rounding error than is already
            // there.
            arr2(&[[16.0, 0.1]]),
        ];
        let fraction = suggest_fraction(data.into_iter(), 16.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 55);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_loss_of_precision() {
        let data = vec![
            // 1/10 is infinite repeating digits in binary, like 1/3 in decimal. Unlike the test
            // just before this one, we do have a loss of precision as the 9 bits needed for the
            // whole number part of 316 will push some bits off the right hand side of the fixed
            // point representation for 0.1.
            arr2(&[[316.0, 0.1]]),
        ];
        let fraction = suggest_fraction(data.into_iter(), 316.0);
        match fraction {
            Round(bits) => {
                assert_eq!(bits, 53);
            }
            _ => {
                assert!(false);
            }
        }
    }
}
