use std::any::TypeId;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::codec::FChunk;
use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite, Serialize};

use super::resolver::Resolver;

const MAGIC_NUMBER: u16 = 0xDCDF + 1;
const FORMAT_VERSION: u32 = 0;

const NODE_SUBCHUNK: u8 = 1;

const TYPE_F32: u8 = 32;
const TYPE_F64: u8 = 64;

/// A non-folder DAG node.
///
pub trait Node<N>: Serialize
where
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8;

    /// Store a node object in the DAG and return its CID
    ///
    fn store(self, resolver: &Arc<Resolver<N>>) -> Result<Cid> {
        Ok(resolver.save_leaf(self)?)
    }

    /// Save a leaf node into the DAG
    ///
    fn save_to(self, stream: &mut impl io::Write) -> Result<()> {
        stream.write_u16(MAGIC_NUMBER)?;
        stream.write_u32(FORMAT_VERSION)?;
        stream.write_byte(Self::type_code())?;
        stream.write_byte(Self::NODE_TYPE)?;
        self.write_to(stream)?;

        Ok(())
    }

    /// Retrieve a node object from the DAG by its CID
    ///
    fn retrieve(resolver: &Arc<Resolver<N>>, cid: &Cid) -> Result<Option<Self>> {
        match resolver.load(cid) {
            Some(mut stream) => Ok(Some(Self::load_from(resolver, &mut stream)?)),
            None => Ok(None),
        }
    }

    /// Load an object from a stream
    fn load_from(_resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        Self::read_header(stream)?;
        Ok(Self::read_from(stream)?)
    }

    /// Read and validate header
    fn read_header(stream: &mut impl io::Read) -> Result<()> {
        let magic_number = stream.read_u16()?;
        if magic_number != MAGIC_NUMBER {
            panic!("File is not a DCDF graph node file.");
        }

        let version = stream.read_u32()?;
        if version != FORMAT_VERSION {
            panic!("Unrecognized file format.");
        }

        if Self::type_code() != stream.read_byte()? {
            panic!("Numeric type doesn't match.");
        }

        let node_type = stream.read_byte()?;
        if Self::NODE_TYPE != node_type {
            panic!("Wrong node type");
        }

        Ok(())
    }

    fn type_code() -> u8 {
        if TypeId::of::<N>() == TypeId::of::<f32>() {
            TYPE_F32
        } else if TypeId::of::<N>() == TypeId::of::<f64>() {
            TYPE_F64
        } else {
            panic!("Unsupported type: {:?}", TypeId::of::<N>())
        }
    }
}

impl<N> Node<N> for FChunk<N>
where
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8 = NODE_SUBCHUNK;
}