//! Functions related to converting floating point numbers to an internal fixed point
//! representation and back again.
//!
//! This is for internal use by DCDF. Number of fractional bits is not stored in the fixed point
//! representation. Instead, that information is stored once for a Chunk, where it is assumed all
//! numbers stored in that Chunk are encoded with the same number of fractional bits.
//!

use num_traits::Float;
use std::fmt::Debug;

/// Convert floating point number to fixed point format
///
/// # Arguments
///
/// * `n` - A floating point number to convert to fixed point
/// * `fractional_bits` - The number of bits to use for the fractional part.
///
/// # Panics
///
/// * When a loss of precision would occur. For lossy conversion see `to_fixed_round`.
/// * When fixed point representation overflows as i64 (ie the number is too big)
/// * When `n` is NaN, inf, or -inf.
///
pub fn to_fixed<F>(n: F, fractional_bits: usize) -> i64
where
    F: Float + Debug,
{
    if !n.is_finite() {
        panic!("Cannot convert {n:?} to fixed point representation.");
    }

    // Shift the point fractional_bits bits to the left
    let shifted = n * F::from(1 << fractional_bits).unwrap();

    // Check to make sure we don't have any bits to the right of the point after shifting
    if shifted.fract() > F::zero() {
        panic!(
            "\
            Converting {:?} to fixed point representation with {} fractional bits \
            results in loss of precision. For lossy conversion you can use `to_fixed_round`. \
        ",
            n, fractional_bits
        );
    }

    match shifted.to_i64() {
        Some(number) => number,
        None => panic!("Overflow converting {n:?} to fixed point representation."),
    }
}

/// Convert floating point number to fixed point format
///
/// Will round do the nearest integer in cases when a loss of precision might occur.
///
/// # Arguments
///
/// * `n` - A floating point number to convert to fixed point
/// * `fractional_bits` - The number of bits to use for the fractional part.
///
/// # Panics
///
/// * When fixed point representation overflows as i64 (ie the number is too big)
/// * When `n` is NaN, inf, or -inf.
///
pub fn to_fixed_round<F>(n: F, fractional_bits: usize) -> i64
where
    F: Float + Debug,
{
    if !n.is_finite() {
        panic!("Cannot convert {n:?} to fixed point representation.");
    }

    // Shift the point fractional_bits bits to the left
    let shifted = n * F::from(1 << fractional_bits).unwrap();

    match shifted.round().to_i64() {
        Some(number) => number,
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
    F::from(n).unwrap() / F::from(1 << fractional_bits).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_fixed() {
        let n = 1.5; // 0b1.1
        assert_eq!(to_fixed(n, 1), 3); // 0b11
        assert_eq!(to_fixed(n, 8), 384); // 0b110000000

        let n = 0.0625; // 0b0.0001
        assert_eq!(to_fixed(n, 4), 1); // 0b1

        let n = 0.0;
        assert_eq!(to_fixed(n, 16), 0);

        let n = -0.0;
        assert_eq!(to_fixed(n, 16), 0);
    }

    #[test]
    fn test_to_fixed_round() {
        let n = 1.5;
        assert_eq!(to_fixed_round(n, 1), 3);
        assert_eq!(to_fixed_round(n, 8), 384);

        let n = 0.0625;
        assert_eq!(to_fixed_round(n, 4), 1);
        assert_eq!(to_fixed_round(n, 3), 1);
        assert_eq!(to_fixed_round(n, 2), 0);

        // 1/10 cannot be precisely represented in binary
        let n = 0.1;
        assert_eq!(to_fixed_round(n, 16), 6554);

        let n = 0.0;
        assert_eq!(to_fixed_round(n, 16), 0);

        let n = -0.0;
        assert_eq!(to_fixed_round(n, 16), 0);
    }

    #[test]
    fn test_from_fixed() {
        assert_eq!(from_fixed::<f32>(3, 1), 1.5);
        assert_eq!(from_fixed::<f64>(384, 8), 1.5);

        assert_eq!(from_fixed::<f32>(1, 4), 0.0625);
        assert_eq!(from_fixed::<f32>(0, 13), 0.0);

        // 1/10 cannot be precisely represented in binary
        assert!(from_fixed::<f64>(6554, 16) - 0.1 < 1e-5);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_loss_of_precision_off_by_1() {
        let n = 0.0625; // 0b0.0001
        to_fixed(n, 3);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_loss_of_precision_off_by_more() {
        let n = 1.0625; // 0b1.0001
        to_fixed(n, 3);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_nan() {
        let n = std::f64::NAN;
        to_fixed(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_inf() {
        let n = std::f64::INFINITY;
        to_fixed(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_neg_inf() {
        let n = std::f64::NEG_INFINITY;
        to_fixed(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_overflow() {
        let n = 1.5e100;
        to_fixed(n, 1);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_round_nan() {
        let n = std::f64::NAN;
        to_fixed_round(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_round_inf() {
        let n = std::f64::INFINITY;
        to_fixed_round(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_round_neg_inf() {
        let n = std::f64::NEG_INFINITY;
        to_fixed_round(n, 14);
    }

    #[test]
    #[should_panic]
    fn test_to_fixed_round_overflow() {
        let n = 1.5e100;
        to_fixed_round(n, 1);
    }
}