use std::{
    fmt::Debug,
    io,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncReadExt};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedRead, ExtendedWrite},
};

use super::{
    node::{AsyncNode, Node, NODE_LINKS},
    resolver::Resolver,
};

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
    N: Float + Debug + Send + Sync + 'static, // # SMELL N is not used
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

#[async_trait]
impl<N> AsyncNode<N> for Links
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Load an object from a stream
    async fn load_from_async(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let n = stream.read_u32_async().await? as usize;

        // Cid doesn't have async read, so read into a buffer and use that
        let mut buffer = vec![];
        stream.read_to_end(&mut buffer).await?;
        let mut stream = io::Cursor::new(buffer);

        let mut links = Vec::with_capacity(n);
        for _ in 0..n {
            // Oh no! No async for this.
            links.push(Cid::read_bytes(&mut stream)?);
        }

        Ok(Self(links))
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
