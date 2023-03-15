use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use cid::Cid;
use futures::io::{AsyncRead, AsyncWrite};
use num_traits::Float;

use crate::{codec::FChunk, errors::Result, extio::Serialize};

use super::resolver::Resolver;

pub(crate) const NODE_LINKS: u8 = 5;
pub(crate) const NODE_MMARRAY3: u8 = 0;
pub(crate) const NODE_SUBCHUNK: u8 = 1;
pub(crate) const NODE_SUPERCHUNK: u8 = 2;
pub(crate) const NODE_SPAN: u8 = 3;

/// A DAG node.
///
#[async_trait]
pub(crate) trait Node<N>: Sized
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()>;

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self>;

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)>;
}

#[async_trait]
impl<N> Node<N> for FChunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_SUBCHUNK;

    async fn load_from(
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        FChunk::read_from(stream).await
    }

    async fn save_to(
        &self,
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        self.write_to(stream).await
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        vec![]
    }
}
