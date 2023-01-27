use std::{fmt::Debug, io, sync::Arc};

use async_trait::async_trait;
use cid::Cid;
use futures::io::AsyncRead;
use num_traits::Float;

use crate::{
    codec::FChunk,
    errors::Result,
    extio::{Serialize, SerializeAsync},
};

use super::resolver::Resolver;

pub(crate) const NODE_COMMIT: u8 = 4;
pub(crate) const NODE_LINKS: u8 = 5;
pub(crate) const NODE_FOLDER: u8 = 3;
pub(crate) const NODE_SUBCHUNK: u8 = 1;
pub(crate) const NODE_SUPERCHUNK: u8 = 2;

/// A DAG node.
///
pub trait Node<N>: Sized
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8;

    /// Save an object into the DAG
    ///
    fn save_to(self, resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()>;

    /// Load an object from a stream
    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self>;

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)>;
}

/// A DAG node.
///
#[async_trait]
pub trait AsyncNode<N>: Node<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Load an object from a stream
    async fn load_from_async(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self>;
}

impl<N> Node<N> for FChunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_SUBCHUNK;

    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()> {
        self.write_to(stream)
    }

    fn load_from(_resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        FChunk::read_from(stream)
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        vec![]
    }
}

#[async_trait]
impl<N> AsyncNode<N> for FChunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    async fn load_from_async(
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        FChunk::read_from_async(stream).await
    }
}
