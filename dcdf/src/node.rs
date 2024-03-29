use std::sync::Arc;

use async_trait::async_trait;
use cid::Cid;
use futures::io::{AsyncRead, AsyncWrite};

use crate::{errors::Result, resolver::Resolver};

pub(crate) const NODE_DATASET: u8 = 0;
pub(crate) const NODE_LINKS: u8 = 1;
pub(crate) const NODE_MMSTRUCT3: u8 = 2;
pub(crate) const NODE_SPAN: u8 = 3;
pub(crate) const NODE_SUBCHUNK: u8 = 4;
pub(crate) const NODE_SUPERCHUNK: u8 = 5;
pub(crate) const NODE_VARIABLE: u8 = 6;

/// A DAG node.
///
#[async_trait]
pub(crate) trait Node: Sized {
    const NODE_TYPE: u8;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()>;

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self>;

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)>;
}
