use cid::Cid;

use super::{mmarray::MMArray1, time::TimeRange};

pub struct Dataset {
    coordinates: [Coordinate; 3],
    variable: Variable,
    shape: [usize; 3],
    resolver: Arc<Resolver<
}

pub struct Coordinate {
    name: String,
    kind: CoordinateKind,
}

pub struct Variable {
    name: String,
    round: Option<usize>,
    span_size: usize,
    k2_levels: Vec<usize>,
    kind: VariableKind,
}

pub enum CoordinateKind {
    Time(TimeRange),
    Array1_f32(Cid),
    Array2_f64(Cid),
}

pub enum VariableKind {
    Array3_f32(Cid),
    Array3_f64(Cid),
}

impl Dataset {
    pub fn new(
        coordinates: [Coordinate; 3],
        shape: [usize; 2],
        time_start: i64,
        time_step: i64,

    ) -> Self {

    }
}