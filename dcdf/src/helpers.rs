use num_traits::Num;
use std::fmt::Debug;

/// Make sure bounds are ordered correctly, eg right is to the right of left, top is above
/// bottom.
///
pub fn rearrange<N>(lower: N, upper: N) -> (N, N)
where
    N: Num + Debug + PartialOrd,
{
    if lower > upper {
        (upper, lower)
    } else {
        (lower, upper)
    }
}
