use std::{fmt::Debug, mem::size_of, sync::Arc};

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncWrite};
use ndarray::Array1;
use num_traits::{cast::cast, Float};

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite},
    node::{Node, NODE_MMARRAY1},
    resolver::Resolver,
};

pub struct Range<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    start: N,
    step: N,
    steps: usize,
}

impl<N> Range<N>
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

impl<N> Cacheable for Range<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        let word_len = size_of::<N>() as u64;

        word_len * 2  // start, step
        + 4 // steps
    }
}

#[async_trait]
impl Node for Range<f32> {
    const NODE_TYPE: u8 = NODE_MMARRAY1;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        _resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_f32(cast(self.start).unwrap()).await?;
        stream.write_f32(cast(self.step).unwrap()).await?;
        stream.write_u32(self.steps as u32).await?;

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        _resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let start = stream.read_f32().await?;
        let step = stream.read_f32().await?;
        let steps = stream.read_u32().await? as usize;

        Ok(Self { start, step, steps })
    }

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)> {
        vec![]
    }
}

// For tests see mmarray
