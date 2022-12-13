use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::codec::FChunk;
use crate::errors::Result;
use crate::extio::Serialize;

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
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8;
    const NODE_TYPE_STR: &'static str;

    /// Save an object into the DAG
    ///
    fn save_to(self, resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()>;

    /// Load an object from a stream
    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self>;

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)> {
        unimplemented!();
    }
}

impl<N> Node<N> for FChunk<N>
where
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8 = NODE_SUBCHUNK;
    const NODE_TYPE_STR: &'static str = "Subchunk";

    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()> {
        self.write_to(stream)
    }

    fn load_from(_resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        FChunk::read_from(stream)
    }
}
