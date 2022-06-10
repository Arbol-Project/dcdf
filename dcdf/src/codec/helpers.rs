use num_traits::PrimInt;
use std::fmt::Debug;

/// Make sure bounds are ordered correctly, eg right is to the right of left, top is above
/// bottom.
pub fn rearrange<I>(lower: I, upper: I) -> (I, I)
where
    I: PrimInt + Debug,
{
    if lower > upper {
        (upper, lower)
    } else {
        (lower, upper)
    }
}
