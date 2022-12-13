use std::fmt::Debug;
use std::io;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::cache::Cacheable;
use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite};

use super::node::{Node, NODE_LINKS};
use super::resolver::Resolver;

pub(crate) struct Links(Vec<Cid>);

impl Links {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
}

impl Deref for Links {
    type Target = Vec<Cid>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Links {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<N> Node<N> for Links
where
    N: Float + Debug + 'static, // # SMELL N is not used
{
    const NODE_TYPE: u8 = NODE_LINKS;
    const NODE_TYPE_STR: &'static str = "Links";

    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()> {
        stream.write_u32(self.0.len() as u32)?;
        for link in &self.0 {
            link.write_bytes(&mut *stream)?;
        }

        Ok(())
    }

    fn load_from(_resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        let n = stream.read_u32()? as usize;
        let mut links = Vec::with_capacity(n);
        for _ in 0..n {
            links.push(Cid::read_bytes(&mut *stream)?);
        }

        Ok(Self(links))
    }
}

impl Cacheable for Links {
    fn size(&self) -> u64 {
        4 + self.0.iter().map(|l| l.to_bytes().len()).sum::<usize>() as u64
    }
}
