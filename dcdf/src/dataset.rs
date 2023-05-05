use std::{cmp, sync::Arc};

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncWrite};
use ndarray::{Array3, ArrayViewMut3, Axis};
use paste::paste;

use crate::{
    cache::Cacheable,
    errors::{Error, Result},
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
    mmarray::{
        MMArray1F32, MMArray1F64, MMArray1I32, MMArray1I64, MMArray3F32, MMArray3F64, MMArray3I32,
        MMArray3I64,
    },
    mmbuffer::MMBuffer3,
    mmstruct::{MMEncoding, MMStruct3},
    node::{Node, NODE_DATASET, NODE_VARIABLE},
    range::{FloatRange, IntRange},
    resolver::Resolver,
    span::Span,
    superchunk::Superchunk,
    time::TimeRange,
};

pub struct Dataset {
    pub coordinates: [Coordinate; 3],
    pub variables: Vec<Variable>,
    pub shape: [usize; 2],

    pub cid: Option<Cid>,
    pub prev: Option<Cid>,

    resolver: Arc<Resolver>,
}

#[derive(Clone)]
pub struct Coordinate {
    pub name: String,
    pub kind: CoordinateKind,
}

#[derive(Clone)]
pub struct Variable {
    /// Name of the variable, e.g. "precipitation"
    pub name: String,

    /// Whether to allow rounding of floating point numbers and, if so, how many bits to allow for
    /// the fraction
    pub round: Option<usize>,

    /// Number of chunks to store per span
    pub span_size: usize,

    /// Number of time instants to store per superchunk
    pub chunk_size: usize,

    /// How to subdivide up superchunks with regard to tree levels
    pub k2_levels: Vec<u32>,

    /// The kind of numerical data stored in this variable
    pub encoding: MMEncoding,

    /// The content identifier for the MMStruct3 structure that provides this variable
    pub cid: Cid,

    resolver: Arc<Resolver>,
}

#[derive(Clone)]
pub enum CoordinateKind {
    Time(TimeRange),
    I32(CoordinateI32),
    I64(CoordinateI64),
    F32(CoordinateF32),
    F64(CoordinateF64),
}

macro_rules! Coordinate {
    ($type:ident) => {
        paste! {
            #[derive(Clone)]
            pub enum [<Coordinate $type>] {
                Range([<MMArray1 $type>]),

                #[allow(dead_code)]
                External(Cid),  // For use in a hypothetical future, for arbitrary mappings
            }
        }
    };
}

Coordinate!(I32);
Coordinate!(I64);
Coordinate!(F32);
Coordinate!(F64);

impl Dataset {
    pub fn new(coordinates: [Coordinate; 3], shape: [usize; 2], resolver: Arc<Resolver>) -> Self {
        Self {
            coordinates,
            shape,
            variables: vec![],
            cid: None,
            prev: None,
            resolver,
        }
    }

    pub async fn commit(&self) -> Result<Cid> {
        self.resolver.save(self).await
    }

    pub async fn add_variable<S: Into<String>>(
        &self,
        name: S,
        round: Option<usize>,
        span_size: usize,
        chunk_size: usize,
        k2_levels: Vec<u32>,
        encoding: MMEncoding,
    ) -> Result<Self> {
        let name = name.into();

        // Initialize an empty span
        let span = Span::new(self.shape, chunk_size, Arc::clone(&self.resolver), encoding);
        let cid = self.resolver.save(&MMStruct3::Span(span)).await?;
        let var = Variable {
            name,
            round,
            span_size,
            chunk_size,
            k2_levels,
            encoding,
            cid,
            resolver: Arc::clone(&self.resolver),
        };

        let mut variables = self.variables.clone();
        variables.push(var);

        let prev = if let Some(cid) = self.cid {
            Some(cid)
        } else {
            self.prev
        };

        Ok(Self {
            variables,
            prev,
            coordinates: self.coordinates.clone(),
            shape: self.shape,
            cid: None,
            resolver: Arc::clone(&self.resolver),
        })
    }

    // SMELL: DRY: use macro for append_XX?

    pub async fn append_i32(&self, name: &str, data: ArrayViewMut3<'_, i32>) -> Result<Self> {
        let variable = self
            .variables
            .iter()
            .filter(|v| v.name == name)
            .next()
            .ok_or(Error::BadName(name.to_string()))?
            .clone();

        // Extract data from last, incomplete span and prepend to passed in data
        let variable = if let Some(tail_data) = variable.tail_data().await? {
            let [instants, rows, cols] = tail_data.shape();
            let mut tail_array = Array3::zeros([instants, rows, cols]);
            let mut tail_buffer = MMBuffer3::new_i32(tail_array.view_mut());
            tail_data
                .fill_window(
                    geom::Cube::new(0, instants, 0, rows, 0, cols),
                    &mut tail_buffer,
                )
                .await?;
            tail_array.append(Axis(0), data.view()).unwrap();

            let buffer = MMBuffer3::new_i32(tail_array.view_mut());
            variable.append(buffer, true).await?
        } else {
            let buffer = MMBuffer3::new_i32(data);
            variable.append(buffer, false).await?
        };

        let mut variables = self.variables.clone();
        for i in 0..variables.len() {
            if variables[i].name == variable.name {
                variables[i] = variable;
                break;
            }
        }

        let prev = if let Some(cid) = self.cid {
            Some(cid)
        } else {
            self.prev
        };

        Ok(Self {
            variables,
            prev,
            coordinates: self.coordinates.clone(),
            shape: self.shape,
            cid: None,
            resolver: Arc::clone(&self.resolver),
        })
    }

    pub async fn append_i64(&self, name: &str, data: ArrayViewMut3<'_, i64>) -> Result<Self> {
        let variable = self
            .variables
            .iter()
            .filter(|v| v.name == name)
            .next()
            .ok_or(Error::BadName(name.to_string()))?
            .clone();

        // Extract data from last, incomplete span and prepend to passed in data
        let variable = if let Some(tail_data) = variable.tail_data().await? {
            let [instants, rows, cols] = tail_data.shape();
            let mut tail_array = Array3::zeros([instants, rows, cols]);
            let mut tail_buffer = MMBuffer3::new_i64(tail_array.view_mut());
            tail_data
                .fill_window(
                    geom::Cube::new(0, instants, 0, rows, 0, cols),
                    &mut tail_buffer,
                )
                .await?;
            tail_array.append(Axis(0), data.view()).unwrap();

            let buffer = MMBuffer3::new_i64(tail_array.view_mut());
            variable.append(buffer, true).await?
        } else {
            let buffer = MMBuffer3::new_i64(data);
            variable.append(buffer, false).await?
        };

        let mut variables = self.variables.clone();
        for i in 0..variables.len() {
            if variables[i].name == variable.name {
                variables[i] = variable;
                break;
            }
        }

        let prev = if let Some(cid) = self.cid {
            Some(cid)
        } else {
            self.prev
        };

        Ok(Self {
            variables,
            prev,
            coordinates: self.coordinates.clone(),
            shape: self.shape,
            cid: None,
            resolver: Arc::clone(&self.resolver),
        })
    }

    pub async fn append_f32(&self, name: &str, data: ArrayViewMut3<'_, f32>) -> Result<Self> {
        let variable = self
            .variables
            .iter()
            .filter(|v| v.name == name)
            .next()
            .ok_or(Error::BadName(name.to_string()))?
            .clone();

        let (round, fractional_bits) = match variable.round {
            Some(bits) => (true, bits),
            None => (false, 0),
        };

        // Extract data from last, incomplete span and prepend to passed in data
        let variable = if let Some(tail_data) = variable.tail_data().await? {
            let [instants, rows, cols] = tail_data.shape();
            let mut tail_array = Array3::zeros([instants, rows, cols]);
            let mut tail_buffer =
                MMBuffer3::new_f32(tail_array.view_mut(), tail_data.fractional_bits(), false);
            tail_data
                .fill_window(
                    geom::Cube::new(0, instants, 0, rows, 0, cols),
                    &mut tail_buffer,
                )
                .await?;
            tail_array.append(Axis(0), data.view()).unwrap();

            let buffer = MMBuffer3::new_f32(tail_array.view_mut(), fractional_bits, round);
            variable.append(buffer, true).await?
        } else {
            let buffer = MMBuffer3::new_f32(data, fractional_bits, round);
            variable.append(buffer, false).await?
        };

        let mut variables = self.variables.clone();
        for i in 0..variables.len() {
            if variables[i].name == variable.name {
                variables[i] = variable;
                break;
            }
        }

        let prev = if let Some(cid) = self.cid {
            Some(cid)
        } else {
            self.prev
        };

        Ok(Self {
            variables,
            prev,
            coordinates: self.coordinates.clone(),
            shape: self.shape,
            cid: None,
            resolver: Arc::clone(&self.resolver),
        })
    }

    pub async fn append_f64(&self, name: &str, data: ArrayViewMut3<'_, f64>) -> Result<Self> {
        let variable = self
            .variables
            .iter()
            .filter(|v| v.name == name)
            .next()
            .ok_or(Error::BadName(name.to_string()))?
            .clone();

        let (round, fractional_bits) = match variable.round {
            Some(bits) => (true, bits),
            None => (false, 0),
        };

        // Extract data from last, incomplete span and prepend to passed in data
        let variable = if let Some(tail_data) = variable.tail_data().await? {
            let [instants, rows, cols] = tail_data.shape();
            let mut tail_array = Array3::zeros([instants, rows, cols]);
            let mut tail_buffer =
                MMBuffer3::new_f64(tail_array.view_mut(), tail_data.fractional_bits(), false);
            tail_data
                .fill_window(
                    geom::Cube::new(0, instants, 0, rows, 0, cols),
                    &mut tail_buffer,
                )
                .await?;
            tail_array.append(Axis(0), data.view()).unwrap();

            let buffer = MMBuffer3::new_f64(tail_array.view_mut(), fractional_bits, round);
            variable.append(buffer, true).await?
        } else {
            let buffer = MMBuffer3::new_f64(data, fractional_bits, round);
            variable.append(buffer, false).await?
        };

        let mut variables = self.variables.clone();
        for i in 0..variables.len() {
            if variables[i].name == variable.name {
                variables[i] = variable;
                break;
            }
        }

        let prev = if let Some(cid) = self.cid {
            Some(cid)
        } else {
            self.prev
        };

        Ok(Self {
            variables,
            prev,
            coordinates: self.coordinates.clone(),
            shape: self.shape,
            cid: None,
            resolver: Arc::clone(&self.resolver),
        })
    }

    pub fn get_coordinate(&self, name: &str) -> Option<&Coordinate> {
        for coord in &self.coordinates {
            if coord.name == name {
                return Some(coord);
            }
        }

        None
    }

    pub fn get_variable(&self, name: &str) -> Option<&Variable> {
        for var in &self.variables {
            if var.name == name {
                return Some(&var);
            }
        }

        None
    }
}

#[async_trait]
impl Node for Dataset {
    const NODE_TYPE: u8 = NODE_DATASET;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        for coord in &self.coordinates {
            coord.write_to(stream).await?;
        }
        stream.write_byte(self.variables.len() as u8).await?;
        for var in &self.variables {
            var.save_to(resolver, stream).await?;
        }
        let [rows, cols] = self.shape;
        stream.write_u32(rows as u32).await?;
        stream.write_u32(cols as u32).await?;
        if let Some(cid) = self.prev {
            stream.write_byte(1).await?;
            stream.write_cid(&cid).await?;
        } else {
            stream.write_byte(0).await?;
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let coordinates = [
            Coordinate::read_from(stream).await?,
            Coordinate::read_from(stream).await?,
            Coordinate::read_from(stream).await?,
        ];
        let n_vars = stream.read_byte().await? as usize;
        let mut variables = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            variables.push(Variable::load_from(resolver, stream).await?);
        }
        let shape = [
            stream.read_u32().await? as usize,
            stream.read_u32().await? as usize,
        ];
        let prev = if stream.read_byte().await? == 1 {
            Some(stream.read_cid().await?)
        } else {
            None
        };

        Ok(Self {
            coordinates,
            variables,
            shape,
            prev,
            cid: None, // FIXME
            resolver: Arc::clone(resolver),
        })
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        let mut ls = vec![];
        for variable in &self.variables {
            ls.push((variable.name.clone(), variable.cid.clone()));
        }
        if let Some(cid) = self.prev {
            ls.push((String::from("prev"), cid.clone()));
        }

        ls
    }
}

impl Cacheable for Dataset {
    fn size(&self) -> u64 {
        let coordinates_size = self.coordinates.iter().map(|c| c.size()).sum::<u64>();
        let variables_size = self.variables.iter().map(|v| v.size()).sum::<u64>();

        Resolver::HEADER_SIZE + coordinates_size + variables_size
            + 8 //shape
            + 1 // prev
            + if let Some(cid) = self.prev { cid.encoded_len() as u64 } else { 0 }
    }
}

impl MMStruct3 {
    fn stride(&self) -> usize {
        match self {
            MMStruct3::Span(span) => span.stride,
            _ => {
                panic!("not a span");
            }
        }
    }

    fn last(&self) -> Cid {
        match self {
            MMStruct3::Span(span) => span.spans[span.spans.len() - 1].clone(),
            _ => {
                panic!("not a span");
            }
        }
    }
    fn len(&self) -> usize {
        match self {
            MMStruct3::Span(span) => span.spans.len(),
            _ => {
                panic!("not a span");
            }
        }
    }

    async fn append(&self, chunk: &MMStruct3) -> Result<Span> {
        match self {
            MMStruct3::Span(span) => span.append(chunk).await,
            _ => {
                panic!("not a span");
            }
        }
    }

    async fn update(&self, chunk: &MMStruct3) -> Result<Span> {
        match self {
            MMStruct3::Span(span) => span.update(chunk).await,
            _ => {
                panic!("not a span");
            }
        }
    }
}

impl Coordinate {
    pub fn data_i32(&self) -> &MMArray1I32 {
        match self.kind {
            CoordinateKind::I32(ref kind) => match kind {
                CoordinateI32::Range(ref range) => &range,
                CoordinateI32::External(_) => {
                    todo!();
                }
            },
            _ => {
                panic!("Not an I32 coordinate");
            }
        }
    }

    pub fn data_i64(&self) -> &MMArray1I64 {
        match self.kind {
            CoordinateKind::I64(ref kind) => match kind {
                CoordinateI64::Range(ref range) => &range,
                CoordinateI64::External(_) => {
                    todo!();
                }
            },
            _ => {
                panic!("Not an I64 coordinate");
            }
        }
    }

    pub fn data_f32(&self) -> &MMArray1F32 {
        match self.kind {
            CoordinateKind::F32(ref kind) => match kind {
                CoordinateF32::Range(ref range) => &range,
                CoordinateF32::External(_) => {
                    todo!();
                }
            },
            _ => {
                panic!("Not an F32 coordinate");
            }
        }
    }

    pub fn data_f64(&self) -> &MMArray1F64 {
        match self.kind {
            CoordinateKind::F64(ref kind) => match kind {
                CoordinateF64::Range(ref range) => &range,
                CoordinateF64::External(_) => {
                    todo!();
                }
            },
            _ => {
                panic!("Not an F64 coordinate");
            }
        }
    }

    pub fn data_time(&self) -> &TimeRange {
        match self.kind {
            CoordinateKind::Time(ref range) => range,
            _ => {
                panic!("Not a time coordinate");
            }
        }
    }

    pub fn len(&self) -> Result<usize> {
        match &self.kind {
            CoordinateKind::Time(_) => Err(Error::TimeIsInfinite),
            CoordinateKind::I32(coord) => match coord {
                CoordinateI32::Range(range) => Ok(range.len()),
                _ => todo!(),
            },
            CoordinateKind::I64(coord) => match coord {
                CoordinateI64::Range(range) => Ok(range.len()),
                _ => todo!(),
            },
            CoordinateKind::F32(coord) => match coord {
                CoordinateF32::Range(range) => Ok(range.len()),
                _ => todo!(),
            },
            CoordinateKind::F64(coord) => match coord {
                CoordinateF64::Range(range) => Ok(range.len()),
                _ => todo!(),
            },
        }
    }

    pub fn range_i32<S: Into<String>>(name: S, start: i32, step: i32, steps: usize) -> Self {
        let name = name.into();
        Self {
            name,
            kind: CoordinateKind::I32(CoordinateI32::Range(MMArray1I32::Range(IntRange::new(
                start, step, steps,
            )))),
        }
    }

    pub fn range_i64<S: Into<String>>(name: S, start: i64, step: i64, steps: usize) -> Self {
        let name = name.into();
        Self {
            name,
            kind: CoordinateKind::I64(CoordinateI64::Range(MMArray1I64::Range(IntRange::new(
                start, step, steps,
            )))),
        }
    }

    pub fn range_f32<S: Into<String>>(name: S, start: f32, step: f32, steps: usize) -> Self {
        let name = name.into();
        Self {
            name,
            kind: CoordinateKind::F32(CoordinateF32::Range(MMArray1F32::Range(FloatRange::new(
                start, step, steps,
            )))),
        }
    }

    pub fn range_f64<S: Into<String>>(name: S, start: f64, step: f64, steps: usize) -> Self {
        let name = name.into();
        Self {
            name,
            kind: CoordinateKind::F64(CoordinateF64::Range(MMArray1F64::Range(FloatRange::new(
                start, step, steps,
            )))),
        }
    }

    pub fn time<S: Into<String>>(name: S, start: i64, step: i64) -> Self {
        let name = name.into();
        Self {
            name,
            kind: CoordinateKind::Time(TimeRange::new(start, step)),
        }
    }
}

#[async_trait]
impl Serialize for Coordinate {
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_str(self.name.as_ref()).await?;
        self.kind.write_to(stream).await?;

        Ok(())
    }

    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let name = stream.read_str().await?;
        let kind = CoordinateKind::read_from(stream).await?;

        Ok(Self { name, kind })
    }
}

impl Cacheable for Coordinate {
    fn size(&self) -> u64 {
        1 + self.name.len() as u64 + self.kind.size()
    }
}

#[async_trait]
impl Serialize for CoordinateKind {
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        match self {
            CoordinateKind::Time(range) => {
                stream.write_byte(MMEncoding::Time as u8).await?;
                stream.write_i64(range.start).await?;
                stream.write_i64(range.step).await?;
            }

            // This code illustrates the kind of ridiculous nesting of enums used for different
            // kinds of coordinates. Maybe think about ways to simplify this structure.
            CoordinateKind::I32(coord) => match coord {
                CoordinateI32::Range(array) => match array {
                    MMArray1I32::Range(range) => {
                        stream.write_byte(MMEncoding::I32 as u8).await?;
                        stream.write_i32(range.start).await?;
                        stream.write_i32(range.step).await?;
                        stream.write_u32(range.steps as u32).await?;
                    }
                },
                _ => {
                    todo!();
                }
            },

            CoordinateKind::I64(coord) => match coord {
                CoordinateI64::Range(array) => match array {
                    MMArray1I64::Range(range) => {
                        stream.write_byte(MMEncoding::I64 as u8).await?;
                        stream.write_i64(range.start).await?;
                        stream.write_i64(range.step).await?;
                        stream.write_u32(range.steps as u32).await?;
                    }
                },
                _ => {
                    todo!();
                }
            },

            CoordinateKind::F32(coord) => match coord {
                CoordinateF32::Range(array) => match array {
                    MMArray1F32::Range(range) => {
                        stream.write_byte(MMEncoding::F32 as u8).await?;
                        stream.write_f32(range.start).await?;
                        stream.write_f32(range.step).await?;
                        stream.write_u32(range.steps as u32).await?;
                    }
                },
                _ => {
                    todo!();
                }
            },

            CoordinateKind::F64(coord) => match coord {
                CoordinateF64::Range(array) => match array {
                    MMArray1F64::Range(range) => {
                        stream.write_byte(MMEncoding::F64 as u8).await?;
                        stream.write_f64(range.start).await?;
                        stream.write_f64(range.step).await?;
                        stream.write_u32(range.steps as u32).await?;
                    }
                },
                _ => {
                    todo!();
                }
            },
        }
        Ok(())
    }

    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let encoding = MMEncoding::try_from(stream.read_byte().await?)?;
        let kind = match encoding {
            MMEncoding::Time => {
                let start = stream.read_i64().await?;
                let step = stream.read_i64().await?;
                CoordinateKind::Time(TimeRange::new(start, step))
            }
            MMEncoding::I32 => {
                let start = stream.read_i32().await?;
                let step = stream.read_i32().await?;
                let steps = stream.read_u32().await? as usize;
                CoordinateKind::I32(CoordinateI32::Range(MMArray1I32::Range(IntRange::new(
                    start, step, steps,
                ))))
            }
            MMEncoding::I64 => {
                let start = stream.read_i64().await?;
                let step = stream.read_i64().await?;
                let steps = stream.read_u32().await? as usize;
                CoordinateKind::I64(CoordinateI64::Range(MMArray1I64::Range(IntRange::new(
                    start, step, steps,
                ))))
            }
            MMEncoding::F32 => {
                let start = stream.read_f32().await?;
                let step = stream.read_f32().await?;
                let steps = stream.read_u32().await? as usize;
                CoordinateKind::F32(CoordinateF32::Range(MMArray1F32::Range(FloatRange::new(
                    start, step, steps,
                ))))
            }
            MMEncoding::F64 => {
                let start = stream.read_f64().await?;
                let step = stream.read_f64().await?;
                let steps = stream.read_u32().await? as usize;
                CoordinateKind::F64(CoordinateF64::Range(MMArray1F64::Range(FloatRange::new(
                    start, step, steps,
                ))))
            }
        };

        Ok(kind)
    }
}

impl Cacheable for CoordinateKind {
    fn size(&self) -> u64 {
        match self {
            CoordinateKind::Time(_) => 16, // start + step
            CoordinateKind::I32(_) => 12,  // start + step + steps
            CoordinateKind::I64(_) => 20,  // start + step + steps
            CoordinateKind::F32(_) => 12,  // start + step + steps
            CoordinateKind::F64(_) => 20,  // start + step + steps
        }
    }
}

impl Variable {
    async fn append(self, mut data: MMBuffer3<'_>, mut update: bool) -> Result<Self> {
        let mut variable = self;
        let mut spans = variable.tail_spans().await?;
        let [instants, rows, cols] = data.shape();
        for start in (0..instants).step_by(variable.chunk_size) {
            // Build next chunk
            let end = cmp::min(start + variable.chunk_size, instants);
            let mut buffer = data.slice(start, end, 0, rows, 0, cols);
            buffer.compute_fractional_bits();
            let chunk = Superchunk::build(
                Arc::clone(&variable.resolver),
                &mut buffer,
                [end - start, rows, cols],
                variable.k2_levels.as_ref(),
                2,
            )
            .await?
            .data;

            // Get span to add chunk to
            let mut span = spans.pop().unwrap();

            // If current tail span is full, save the current span and create a new, open span
            if span.shape()[0] == variable.span_size * span.stride() {
                spans.push(span);
                variable = variable.save_spans(spans).await?;
                variable = variable.create_open_span([rows, cols]).await?;
                spans = variable.tail_spans().await?;
                span = spans.pop().unwrap();
                assert_eq!(span.len(), 0);
            }

            // Add chunk to span
            let span = if update {
                update = false;
                span.update(&chunk).await?
            } else {
                span.append(&chunk).await?
            };
            spans.push(Arc::new(MMStruct3::Span(span)));
        }

        // Done, save spans
        variable.save_spans(spans).await
    }

    async fn create_open_span(self, shape: [usize; 2]) -> Result<Self> {
        // Create a new bottom level span for adding chunks to
        let mut span = Span::new(
            shape,
            self.chunk_size,
            Arc::clone(&self.resolver),
            self.encoding,
        );

        // Find a place to add it to the tree
        let mut spans = self.tail_spans().await?;
        let mut left_hand = spans.pop().unwrap();

        loop {
            if let Some(parent) = spans.pop() {
                if parent.len() == self.span_size {
                    // This node is also full. Create a new parent for the new span, then try
                    // another level up
                    let new_parent = Span::new(
                        shape,
                        self.span_size * span.stride,
                        Arc::clone(&self.resolver),
                        self.encoding,
                    );
                    left_hand = parent;
                    span = new_parent.append(&MMStruct3::Span(span)).await?;
                } else {
                    // Found a node with an opening, add the new span there
                    span = parent.append(&MMStruct3::Span(span)).await?;
                    break;
                }
            } else {
                // No ancestor nodes have any room. Create a new root and move existing root down
                // one level
                let mut new_root = Span::new(
                    shape,
                    self.span_size * span.stride,
                    Arc::clone(&self.resolver),
                    self.encoding,
                );
                let right_hand = span;
                new_root = new_root.append(&*left_hand).await?;
                span = new_root.append(&MMStruct3::Span(right_hand)).await?;
                break;
            }
        }

        // Update any ancestor nodes
        while let Some(ancestor) = spans.pop() {
            span = ancestor.update(&MMStruct3::Span(span)).await?;
        }

        let cid = self.resolver.save(&MMStruct3::Span(span)).await?;

        Ok(Self { cid, ..self })
    }

    async fn tail_data(&self) -> Result<Option<Arc<MMStruct3>>> {
        // Get last span
        let tail_spans = self.tail_spans().await?;
        let tail = &tail_spans[tail_spans.len() - 1];

        // Special case, no chunks yet
        if tail.len() == 0 {
            return Ok(None);
        }

        // Get last chunk of last span
        let cid = tail.last();
        let chunk = self.resolver.get_mmstruct3(&cid).await?;

        if chunk.shape()[0] < self.chunk_size {
            Ok(Some(chunk))
        } else {
            // Last chunk is full, no need to do anything with it
            Ok(None)
        }
    }

    /// Returns all spans traversed in the tree to get to the last span in the dataset.
    ///
    async fn tail_spans(&self) -> Result<Vec<Arc<MMStruct3>>> {
        // Find lowest, rightmost span
        let mut ancestors = vec![];
        let mut span = self.resolver.get_mmstruct3(&self.cid).await?;

        while span.stride() > self.chunk_size {
            let cid = span.last();
            ancestors.push(span);
            span = self.resolver.get_mmstruct3(&cid).await?;
        }

        ancestors.push(span);

        Ok(ancestors)
    }

    async fn save_spans(self, mut spans: Vec<Arc<MMStruct3>>) -> Result<Self> {
        let mut span = spans.pop().unwrap();
        while let Some(last) = spans.pop() {
            span = Arc::new(MMStruct3::Span(last.update(&span).await?));
        }

        let cid = self.resolver.save(&*span).await?;

        Ok(Self { cid, ..self })
    }

    pub async fn data_i32(&self) -> Result<MMArray3I32> {
        Ok(MMArray3I32::new(
            self.resolver.get_mmstruct3(&self.cid).await?,
        ))
    }

    pub async fn data_i64(&self) -> Result<MMArray3I64> {
        Ok(MMArray3I64::new(
            self.resolver.get_mmstruct3(&self.cid).await?,
        ))
    }

    pub async fn data_f32(&self) -> Result<MMArray3F32> {
        Ok(MMArray3F32::new(
            self.resolver.get_mmstruct3(&self.cid).await?,
        ))
    }

    pub async fn data_f64(&self) -> Result<MMArray3F64> {
        Ok(MMArray3F64::new(
            self.resolver.get_mmstruct3(&self.cid).await?,
        ))
    }
}

#[async_trait]
impl Node for Variable {
    const NODE_TYPE: u8 = NODE_VARIABLE;

    async fn save_to(
        &self,
        _resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_str(self.name.as_ref()).await?;
        if let Some(bits) = self.round {
            stream.write_byte(1).await?;
            stream.write_byte(bits as u8).await?;
        } else {
            stream.write_byte(0).await?;
        }
        stream.write_u32(self.span_size as u32).await?;
        stream.write_u32(self.chunk_size as u32).await?;
        stream.write_byte(self.k2_levels.len() as u8).await?;
        for levels in &self.k2_levels {
            stream.write_u32(*levels).await?;
        }
        stream.write_byte(self.encoding as u8).await?;
        stream.write_cid(&self.cid).await?;

        Ok(())
    }

    async fn load_from(
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let name = stream.read_str().await?;
        let round = if stream.read_byte().await? == 1 {
            Some(stream.read_byte().await? as usize)
        } else {
            None
        };
        let span_size = stream.read_u32().await? as usize;
        let chunk_size = stream.read_u32().await? as usize;
        let n_k2_levels = stream.read_byte().await? as usize;
        let mut k2_levels = Vec::with_capacity(n_k2_levels);
        for _ in 0..n_k2_levels {
            k2_levels.push(stream.read_u32().await?);
        }
        let encoding = MMEncoding::try_from(stream.read_byte().await?)?;
        let cid = stream.read_cid().await?;

        Ok(Self {
            name,
            round,
            span_size,
            chunk_size,
            k2_levels,
            encoding,
            cid,
            resolver: Arc::clone(resolver),
        })
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        // Won't be called. Variable isn't really a DAG Node, it just needs the resolver when
        // loading itself from a stream, so we implemented Node instead of Serialize
        unimplemented!();
    }
}

impl Cacheable for Variable {
    fn size(&self) -> u64 {
        1 + self.name.len() as u64
        + 1 + if self.round.is_some() { 1 } else { 0 }
        + 4 // span size
        + 4 // chunk size
        + 4 * self.k2_levels.len() as u64
        + 1 // encoding
        + self.cid.encoded_len() as u64
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{s, Array3};
    use num_traits::{cast, Float, PrimInt};

    use super::*;

    use crate::{
        fixed::{from_fixed, to_fixed},
        geom, testing,
    };

    macro_rules! constructor_tests {
        ($type:ty) => {
            paste! {

                fn [<make_one_ $type>](resolver: Arc<Resolver>) -> Dataset {
                    let n = |n| n as $type;
                    let t = Coordinate::time("t", 0, 100);
                    let y = Coordinate::[<range_ $type>]("y", n(-160), n(20), 16);
                    let x = Coordinate::[<range_ $type>]("x", n(-200), n(25), 16);
                    Dataset::new([t, y, x], [16, 16], resolver)
                }

                #[tokio::test]
                async fn [<test_new_ $type>]() -> Result<()> {
                    let n = |n| n as $type;
                    let resolver = testing::resolver();
                    let dataset = [<make_one_ $type>](resolver);

                    assert_eq!(dataset.coordinates[0].name, "t");
                    assert!(dataset.coordinates[0].len().is_err());
                    assert_eq!(dataset.coordinates[1].name, "y");
                    assert_eq!(dataset.coordinates[1].len().unwrap(), 16);
                    assert_eq!(dataset.coordinates[2].name, "x");
                    assert_eq!(dataset.coordinates[2].len().unwrap(), 16);

                    assert_eq!(dataset.shape, [16, 16]);
                    assert_eq!(dataset.prev, None);
                    assert_eq!(dataset.cid, None);

                    assert_eq!(dataset.variables.len(), 0);

                    let t = dataset.get_coordinate("t").unwrap().data_time();
                    assert_eq!(t.get(10), 1000);

                    let y = dataset.get_coordinate("y").unwrap().[<data_ $type>]();
                    assert_eq!(y.get(10), n(40));

                    let x = dataset.get_coordinate("x").unwrap().[<data_ $type>]();
                    assert_eq!(x.get(10), n(50));

                    assert!(dataset.get_coordinate("doesn't exist").is_none());
                    assert!(dataset.get_variable("also doesn't exist").is_none());

                    Ok(())
                }
            }
        };
    }

    constructor_tests!(i32);
    constructor_tests!(i64);
    constructor_tests!(f32);
    constructor_tests!(f64);

    fn make_int_data<N: PrimInt>(instants: usize) -> Array3<N> {
        let data = testing::array(8);
        let data = Array3::from_shape_fn([instants, 16, 16], |(t, y, x)| {
            cast(data[[t % 100, y % 8, x % 8]]).unwrap()
        });

        data
    }

    fn make_float_data<N: Float>(instants: usize) -> Array3<N> {
        let data = testing::farray(8);
        let data = Array3::from_shape_fn([instants, 16, 16], |(t, y, x)| {
            cast(data[[t % 100, y % 8, x % 8]]).unwrap()
        });

        data
    }

    async fn populate(
        dataset: Dataset,
    ) -> Result<(
        Array3<f32>,
        Array3<f64>,
        Array3<i32>,
        Array3<i64>,
        Array3<f32>,
        Array3<f64>,
        Dataset,
    )> {
        assert!(dataset.cid.is_none());
        assert_eq!(dataset.ls().len(), 0);

        let dataset = dataset
            .add_variable("apples", None, 10, 20, vec![2, 2], MMEncoding::F32)
            .await?;
        let mut apple_data = make_float_data::<f32>(360);
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![..99_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![99..200_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![200_usize.., .., ..]))
            .await?;

        let dataset = dataset
            .add_variable("pears", None, 10, 20, vec![2, 2], MMEncoding::F64)
            .await?;
        let mut pear_data = make_float_data::<f64>(500);
        let dataset = dataset
            .append_f64("pears", pear_data.slice_mut(s![..189_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f64("pears", pear_data.slice_mut(s![189..400_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f64("pears", pear_data.slice_mut(s![400_usize.., .., ..]))
            .await?;

        let dataset = dataset
            .add_variable("bananas", None, 10, 20, vec![2, 2], MMEncoding::I32)
            .await?;
        let mut banana_data = make_int_data::<i32>(511);
        let dataset = dataset
            .append_i32("bananas", banana_data.slice_mut(s![..59_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_i32("bananas", banana_data.slice_mut(s![59..300_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_i32("bananas", banana_data.slice_mut(s![300_usize.., .., ..]))
            .await?;

        let dataset = dataset
            .add_variable("grapes", None, 10, 20, vec![2, 2], MMEncoding::I64)
            .await?;
        let mut grape_data = make_int_data::<i64>(3650);
        let dataset = dataset
            .append_i64("grapes", grape_data.slice_mut(s![..1079_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_i64("grapes", grape_data.slice_mut(s![1079_usize..3000, .., ..]))
            .await?;
        let dataset = dataset
            .append_i64("grapes", grape_data.slice_mut(s![3000_usize.., .., ..]))
            .await?;

        assert!(dataset.prev.is_none());
        assert_eq!(dataset.ls().len(), 4);
        let resolver = Arc::clone(&dataset.resolver);
        let cid = dataset.commit().await?;
        let dataset = resolver.get_dataset(&cid).await?;
        assert_eq!(dataset.cid, Some(cid));

        let dataset = dataset
            .add_variable("dates", Some(2), 10, 20, vec![2, 2], MMEncoding::F32)
            .await?;

        let mut date_data = make_float_data::<f32>(489);
        let dataset = dataset.append_f32("dates", date_data.view_mut()).await?;
        date_data.mapv_inplace(|v| from_fixed(to_fixed(v, 2, true), 2));

        assert!(dataset.cid.is_none());
        assert_eq!(dataset.prev, Some(cid));

        let dataset = dataset
            .add_variable("melons", Some(2), 10, 20, vec![2, 2], MMEncoding::F64)
            .await?;
        let mut melon_data = make_float_data::<f64>(275);
        let dataset = dataset.append_f64("melons", melon_data.view_mut()).await?;
        melon_data.mapv_inplace(|v| from_fixed(to_fixed(v, 2, true), 2));

        assert!(dataset.cid.is_none());
        assert_eq!(dataset.prev, Some(cid));

        assert_eq!(dataset.ls().len(), 7);
        assert_eq!(dataset.ls()[6].0, String::from("prev"));

        Ok((
            apple_data,
            pear_data,
            banana_data,
            grape_data,
            date_data,
            melon_data,
            dataset,
        ))
    }

    async fn verify_i32(array: Array3<i32>, mmarray: MMArray3I32) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_i64(array: Array3<i64>, mmarray: MMArray3I64) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_f32(array: Array3<f32>, mmarray: MMArray3F32) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_f64(array: Array3<f64>, mmarray: MMArray3F64) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    #[tokio::test]
    async fn test_populate_variables_and_save_load() -> Result<()> {
        let resolver = testing::resolver();
        let dataset = make_one_f64(Arc::clone(&resolver));
        let (apple_data, pear_data, banana_data, grape_data, date_data, melon_data, dataset) =
            populate(dataset).await?;
        assert_eq!(dataset.variables.len(), 6);

        let cid = dataset.commit().await?;
        let dataset = resolver.get_dataset(&cid).await?;

        let apples = dataset.get_variable("apples").unwrap();
        assert_eq!(apples.name, "apples");
        assert!(apples.round.is_none());
        assert_eq!(apples.span_size, 10);
        assert_eq!(apples.chunk_size, 20);
        assert_eq!(apples.k2_levels, &[2, 2]);
        assert_eq!(apples.encoding, MMEncoding::F32);

        let apples_mmstruct = apples.data_f32().await?;
        verify_f32(apple_data, apples_mmstruct).await?;

        let pears = dataset.get_variable("pears").unwrap();
        assert_eq!(pears.name, "pears");
        assert!(pears.round.is_none());
        assert_eq!(pears.span_size, 10);
        assert_eq!(pears.chunk_size, 20);
        assert_eq!(pears.k2_levels, &[2, 2]);
        assert_eq!(pears.encoding, MMEncoding::F64);

        let pears_mmstruct = pears.data_f64().await?;
        verify_f64(pear_data, pears_mmstruct).await?;

        let bananas = dataset.get_variable("bananas").unwrap();
        assert_eq!(bananas.name, "bananas");
        assert!(bananas.round.is_none());
        assert_eq!(bananas.span_size, 10);
        assert_eq!(bananas.chunk_size, 20);
        assert_eq!(bananas.k2_levels, &[2, 2]);
        assert_eq!(bananas.encoding, MMEncoding::I32);

        let bananas_mmstruct = bananas.data_i32().await?;
        verify_i32(banana_data, bananas_mmstruct).await?;

        let grapes = dataset.get_variable("grapes").unwrap();
        assert_eq!(grapes.name, "grapes");
        assert!(grapes.round.is_none());
        assert_eq!(grapes.span_size, 10);
        assert_eq!(grapes.chunk_size, 20);
        assert_eq!(grapes.k2_levels, &[2, 2]);
        assert_eq!(grapes.encoding, MMEncoding::I64);

        let grapes_mmstruct = grapes.data_i64().await?;
        verify_i64(grape_data, grapes_mmstruct).await?;

        let dates = dataset.get_variable("dates").unwrap();
        assert_eq!(dates.name, "dates");
        assert_eq!(dates.round, Some(2));
        assert_eq!(dates.span_size, 10);
        assert_eq!(dates.chunk_size, 20);
        assert_eq!(dates.k2_levels, &[2, 2]);
        assert_eq!(dates.encoding, MMEncoding::F32);

        let dates_mmstruct = dates.data_f32().await?;
        verify_f32(date_data, dates_mmstruct).await?;

        let melons = dataset.get_variable("melons").unwrap();
        assert_eq!(melons.name, "melons");
        assert_eq!(melons.round, Some(2));
        assert_eq!(melons.span_size, 10);
        assert_eq!(melons.chunk_size, 20);
        assert_eq!(melons.k2_levels, &[2, 2]);
        assert_eq!(melons.encoding, MMEncoding::F64);

        let melons_mmstruct = melons.data_f64().await?;
        verify_f64(melon_data, melons_mmstruct).await?;

        let ls = resolver.ls(&cid).await?;
        assert_eq!(ls.len(), 7);

        assert_eq!(ls[0].name, String::from("apples"));
        assert_eq!(ls[0].cid, apples.cid);
        assert_eq!(ls[0].node_type, Some("Span"));

        assert_eq!(ls[1].name, String::from("pears"));
        assert_eq!(ls[1].cid, pears.cid);
        assert_eq!(ls[1].node_type, Some("Span"));

        assert_eq!(ls[2].name, String::from("bananas"));
        assert_eq!(ls[2].cid, bananas.cid);
        assert_eq!(ls[2].node_type, Some("Span"));

        assert_eq!(ls[3].name, String::from("grapes"));
        assert_eq!(ls[3].cid, grapes.cid);
        assert_eq!(ls[3].node_type, Some("Span"));

        assert_eq!(ls[4].name, String::from("dates"));
        assert_eq!(ls[4].cid, dates.cid);
        assert_eq!(ls[4].node_type, Some("Span"));

        assert_eq!(ls[5].name, String::from("melons"));
        assert_eq!(ls[5].cid, melons.cid);
        assert_eq!(ls[5].node_type, Some("Span"));

        assert_eq!(ls[6].name, String::from("prev"));
        assert_eq!(ls[6].cid, dataset.prev.unwrap());
        assert_eq!(ls[6].node_type, Some("Dataset"));

        Ok(())
    }
}
