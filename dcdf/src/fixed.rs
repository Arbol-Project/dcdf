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
pub(crate) fn to_fixed<F>(n: F, fractional_bits: usize, round: bool) -> i64
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
        }
        else {
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
        Some(number) => number + 1,  // Set LSB to 1 to indicate not a NaN
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
pub(crate) fn from_fixed<F: Float>(n: i64, fractional_bits: usize) -> F {
    match n {
        0 => F::nan(),
        _ => F::from(n - 1).unwrap() / F::from(1 << (fractional_bits + 1)).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_fixed() {
        let n = 1.5; // 0b1.1
        assert_eq!(to_fixed(n, 1, false), 7); // 0b111
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
}
