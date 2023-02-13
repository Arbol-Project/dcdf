use std::{
    collections::BTreeMap,
    fmt::Debug,
    io::{Read, Write},
    sync::Arc,
};

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite},
};

use super::{
    node::{AsyncNode, Node, NODE_FOLDER},
    resolver::Resolver,
};

pub struct Folder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    items: BTreeMap<String, Cid>,
    resolver: Arc<Resolver<N>>,
}

impl<N> Folder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn new(resolver: &Arc<Resolver<N>>) -> Self {
        Self {
            items: BTreeMap::new(),
            resolver: Arc::clone(resolver),
        }
    }

    pub fn update<S>(&self, name: S, object: Cid) -> Self
    where
        S: Into<String>,
    {
        let mut items = self.items.clone();
        let name = name.into();
        items.insert(name, object);

        Self {
            items,
            resolver: Arc::clone(&self.resolver),
        }
    }

    pub fn insert<O, S>(&self, name: S, object: O) -> Result<Self>
    where
        O: Node<N>,
        S: Into<String>,
    {
        let object = self.resolver.save(object)?;

        Ok(self.update(name, object))
    }

    pub async fn insert_async<O, S>(&self, name: S, object: O) -> Result<Self>
    where
        O: AsyncNode<N>,
        S: Into<String>,
    {
        let object = self.resolver.save_async(object).await?;

        Ok(self.update(name, object))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Cid)> {
        self.items.iter()
    }

    pub fn get<S>(&self, key: S) -> Option<Cid>
    where
        S: Into<String>,
    {
        let key = key.into();
        self.items.get(&key).and_then(|cid| Some(cid.clone()))
    }
}

impl<N> Node<N> for Folder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_FOLDER;

    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl Read) -> Result<Self> {
        let mut items = BTreeMap::new();
        let n_items = stream.read_u32()? as usize;
        for _ in 0..n_items {
            let strlen = stream.read_byte()? as u64;
            let mut key = String::with_capacity(strlen as usize);
            stream.take(strlen).read_to_string(&mut key)?;

            let item = Cid::read_bytes(&mut *stream)?;
            items.insert(key, item);
        }

        Ok(Self {
            items,
            resolver: Arc::clone(resolver),
        })
    }

    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl Write) -> Result<()> {
        stream.write_u32(self.items.len() as u32)?;
        for (key, value) in &self.items {
            stream.write_byte(key.len() as u8)?;
            stream.write_all(key.as_bytes())?;
            value.write_bytes(&mut *stream)?;
        }

        Ok(())
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        let mut ls = Vec::new();
        for (name, cid) in &self.items {
            ls.push((name.clone(), cid.clone()));
        }

        ls
    }
}

#[async_trait]
impl<N> AsyncNode<N> for Folder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Save an object into the DAG
    ///
    async fn save_to_async(
        self,
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_u32_async(self.items.len() as u32).await?;
        for (key, value) in &self.items {
            stream.write_byte_async(key.len() as u8).await?;
            stream.write_all(key.as_bytes()).await?;
            stream.write_cid(&value).await?;
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from_async(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let mut items = BTreeMap::new();
        let n_items = stream.read_u32_async().await? as usize;
        for _ in 0..n_items {
            let strlen = stream.read_byte_async().await? as u64;
            let mut key = String::with_capacity(strlen as usize);
            stream.take(strlen).read_to_string(&mut key).await?;

            let item = stream.read_cid().await?;
            items.insert(key, item);
        }

        Ok(Self {
            items,
            resolver: Arc::clone(resolver),
        })
    }
}

impl<N> Cacheable for Folder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        Resolver::<N>::HEADER_SIZE
            + 4
            + self
                .items
                .iter()
                .map(|(key, value)| 1 + key.len() + value.encoded_len())
                .sum::<usize>() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cid::Cid;

    use crate::errors::Result;

    use super::super::testing;

    fn folders(resolver: &Arc<Resolver<f32>>) -> Result<Cid> {
        let data1 = testing::array(16);
        let superchunk1 = testing::superchunk(&data1, resolver)?;
        let superchunk1 = resolver.save(superchunk1)?;

        let a = Folder::new(resolver);
        let a = a.update("data", superchunk1);

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, resolver)?;
        let superchunk2 = resolver.save(superchunk2)?;

        let b = Folder::new(resolver);
        let b = b.update("data", superchunk2);

        let c = Folder::new(resolver);
        let c = c.update("a", resolver.save(a)?);
        let c = c.update("b", resolver.save(b)?);

        resolver.save(c)
    }

    async fn folders_async(resolver: &Arc<Resolver<f32>>) -> Result<Cid> {
        let data1 = testing::array(16);
        let superchunk1 = testing::superchunk(&data1, resolver)?;
        let superchunk1 = resolver.save_async(superchunk1).await?;

        let a = Folder::new(resolver);
        let a = a.update("data", superchunk1);

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, resolver)?;
        let superchunk2 = resolver.save_async(superchunk2).await?;

        let b = Folder::new(resolver);
        let b = b.update("data", superchunk2);

        let c = Folder::new(resolver);
        let c = c.update("a", resolver.save_async(a).await?);
        let c = c.update("b", resolver.save_async(b).await?);

        resolver.save_async(c).await
    }

    #[test]
    fn test_iter() -> Result<()> {
        let resolver = testing::resolver();
        let cid = folders(&resolver)?;
        let c = resolver.get_folder(&cid)?;

        let mut contents = c.iter();
        let (key, cid) = contents.next().expect("Expecting folder a");
        assert_eq!(key, "a");
        let a = resolver.get_folder(&cid)?;

        let (key, cid) = contents.next().expect("Expecting folder b");
        assert_eq!(key, "b");
        let b = resolver.get_folder(&cid)?;

        assert!(contents.next().is_none());

        let mut contents = a.iter();
        let (key, cid) = contents.next().expect("Expecting data superchunk");
        assert_eq!(key, "data");
        let superchunk = resolver.get_superchunk(&cid)?;

        assert_eq!(superchunk.shape(), [100, 16, 16]);
        assert!(contents.next().is_none());

        let mut contents = b.iter();
        let (key, cid) = contents.next().expect("Expecting data superchunk");
        assert_eq!(key, "data");
        let superchunk = resolver.get_superchunk(&cid)?;

        assert_eq!(superchunk.shape(), [100, 15, 15]);
        assert!(contents.next().is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_get() -> Result<()> {
        let resolver = testing::resolver();
        let cid = folders_async(&resolver).await?;
        let c = resolver.get_folder_async(&cid).await?;

        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder_async(&a).await?;

        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder_async(&b).await?;

        let superchunk = a.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk_async(&superchunk).await?;

        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = b.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk_async(&superchunk).await?;

        assert_eq!(superchunk.shape(), [100, 15, 15]);

        Ok(())
    }

    #[test]
    fn test_ls() -> Result<()> {
        let resolver = testing::resolver();
        let cid = folders(&resolver)?;
        let c = resolver.get_folder(&cid)?;

        let ls = c.ls();
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0], (String::from("a"), c.get("a").unwrap()));
        let a = resolver.get_folder(&ls[0].1)?;
        assert_eq!(ls[1], (String::from("b"), c.get("b").unwrap()));
        let b = resolver.get_folder(&ls[1].1)?;

        let ls = a.ls();
        assert_eq!(ls, vec![(String::from("data"), a.get("data").unwrap())]);

        let ls = b.ls();
        assert_eq!(ls, vec![(String::from("data"), b.get("data").unwrap())]);

        Ok(())
    }
}
