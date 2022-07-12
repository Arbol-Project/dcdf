use cid::Cid;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::rc::{Rc, Weak};

use super::mapper::Mapper;
use crate::codec::FChunk;

pub struct Resolver<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    mapper: M,
    chunks: HashMap<Cid, Weak<FChunk<N>>>, // TODO use LRU cache
}

impl<M, N> Resolver<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    pub fn new(mapper: M) -> Self {
        Self {
            mapper,
            chunks: HashMap::new(),
        }
    }

    pub fn get_chunk(&mut self, cid: Cid) -> io::Result<Rc<FChunk<N>>> {
        let chunk = match self.chunks.get(&cid) {
            Some(reference) => match reference.upgrade() {
                Some(reference) => Some(reference),
                None => None,
            },
            None => None,
        };

        let reference = match chunk {
            Some(reference) => reference,
            None => {
                let mut reader = self
                    .mapper
                    .load(cid)
                    .expect("Can't find chunk for CID {cid:?}");
                let chunk = FChunk::deserialize(&mut reader)?;
                let reference = Rc::new(chunk);
                self.chunks.insert(cid, Rc::downgrade(&reference));

                reference
            }
        };

        Ok(reference)
    }
}
