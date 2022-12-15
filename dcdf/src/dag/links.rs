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

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)> {
        let mut ls = Vec::new();
        for (i, cid) in self.iter().enumerate() {
            ls.push((i.to_string(), cid.clone()));
        }

        ls
    }
}

impl Cacheable for Links {
    fn size(&self) -> u64 {
        Resolver::<f32>::HEADER_SIZE
            + 4
            + self.0.iter().map(|l| l.to_bytes().len()).sum::<usize>() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::testing;

    fn make_one() -> Links {
        let mut links = Links::new();
        links.push(testing::cid_for("zero"));
        links.push(testing::cid_for("one"));
        links.push(testing::cid_for("two"));

        links
    }

    #[test]
    fn serialize_deserialize() -> Result<()> {
        let resolver: Arc<Resolver<f32>> = testing::resolver();
        let links = make_one();
        let expected = links.0.clone();

        let cid = resolver.save(links)?;
        let links = resolver.get_links(&cid)?;
        assert_eq!(expected, links.0);

        Ok(())
    }

    #[test]
    fn ls() {
        let links = make_one();
        let ls = <Links as Node<f32>>::ls(&links);
        assert_eq!(ls[0], (String::from("0"), testing::cid_for("zero")));
        assert_eq!(ls[1], (String::from("1"), testing::cid_for("one")));
        assert_eq!(ls[2], (String::from("2"), testing::cid_for("two")));
    }
}
