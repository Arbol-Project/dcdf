use ndarray::Array1;

#[derive(Clone)]
pub struct TimeRange {
    pub start: i64,
    pub step: i64,
}

impl TimeRange {
    pub fn new(start: i64, step: i64) -> Self {
        Self { start, step }
    }

    pub fn get(&self, index: usize) -> i64 {
        self.start + (index as i64) * self.step
    }

    pub fn slice(&self, start: usize, stop: usize) -> Array1<i64> {
        Array1::from_iter((start..stop).map(|i| self.start + (i as i64) * self.step))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_get() {
        let range = TimeRange::new(1000000, 3600);
        assert_eq!(range.get(0), 1000000);
        assert_eq!(range.get(100), 1360000);
    }

    #[test]
    fn test_slice() {
        let range = TimeRange::new(1000000, 3600);
        assert_eq!(range.slice(100, 102), array![1360000, 1363600]);
    }
}
