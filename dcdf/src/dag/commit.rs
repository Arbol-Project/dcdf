use std::{fmt::Debug, io, sync::Arc};

use async_trait::async_trait;
use cid::Cid;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite},
};

use super::{
    folder::Folder,
    node::{AsyncNode, Node, NODE_COMMIT},
    resolver::Resolver,
};

pub struct Commit<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    message: String,
    pub prev: Option<Cid>,
    pub root: Cid,

    resolver: Arc<Resolver<N>>,
}

impl<N> Commit<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn new<S>(message: S, root: Cid, prev: Option<Cid>, resolver: &Arc<Resolver<N>>) -> Self
    where
        S: Into<String>,
    {
        let message = message.into();
        Self {
            message,
            prev,
            root,

            resolver: Arc::clone(resolver),
        }
    }

    pub fn message(&self) -> &str {
        self.message.as_ref()
    }

    pub fn prev(&self) -> Result<Option<Arc<Self>>> {
        match self.prev {
            Some(cid) => Ok(Some(self.resolver.get_commit(&cid)?)),
            None => Ok(None),
        }
    }

    pub async fn prev_async(&self) -> Result<Option<Arc<Self>>> {
        match self.prev {
            Some(cid) => Ok(Some(self.resolver.get_commit_async(&cid).await?)),
            None => Ok(None),
        }
    }

    pub fn root(&self) -> Arc<Folder<N>> {
        self.resolver
            .get_folder(&self.root)
            .expect("Root not found")
    }

    pub async fn root_async(&self) -> Arc<Folder<N>> {
        self.resolver
            .get_folder_async(&self.root)
            .await
            .expect("Root not found")
    }
}

impl<N> Node<N> for Commit<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_COMMIT;

    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        let root = Cid::read_bytes(&mut *stream)?;
        let prev = match stream.read_byte()? {
            0 => None,
            _ => Some(Cid::read_bytes(&mut *stream)?),
        };
        let mut message = String::new();
        stream.read_to_string(&mut message)?;

        Ok(Self {
            root,
            prev,
            message,

            resolver: Arc::clone(resolver),
        })
    }

    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()> {
        self.root.write_bytes(&mut *stream)?;
        match self.prev {
            Some(cid) => {
                stream.write_byte(1)?;
                cid.write_bytes(&mut *stream)?;
            }
            None => {
                stream.write_byte(0)?;
            }
        }
        stream.write_all(self.message.as_bytes())?;

        Ok(())
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        let mut ls = vec![(String::from("root"), self.root)];
        if let Some(prev) = self.prev {
            ls.push((String::from("prev"), prev));
        }

        ls
    }
}

#[async_trait]
impl<N> AsyncNode<N> for Commit<N>
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
        stream.write_cid(&self.root).await?;
        match self.prev {
            Some(cid) => {
                stream.write_byte_async(1).await?;
                stream.write_cid(&cid).await?;
            }
            None => {
                stream.write_byte_async(0).await?;
            }
        }
        stream.write_all(self.message.as_bytes()).await?;

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from_async(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let root = stream.read_cid().await?;
        let prev = match stream.read_byte_async().await? {
            0 => None,
            _ => Some(stream.read_cid().await?),
        };
        let mut message = String::new();
        stream.read_to_string(&mut message).await?;

        Ok(Self {
            root,
            prev,
            message,

            resolver: Arc::clone(resolver),
        })
    }
}

impl<N> Cacheable for Commit<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        Resolver::<N>::HEADER_SIZE
            + (self.root.encoded_len()
                + 1
                + match self.prev {
                    Some(cid) => cid.encoded_len(),
                    None => 0,
                }
                + self.message.len()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::testing;

    #[test]
    fn make_a_couple_of_commits() -> Result<()> {
        // Store DAG structure
        let resolver = testing::resolver();
        let data1 = testing::array(16);
        let superchunk1 = testing::superchunk(&data1, &resolver)?;

        let a = Folder::new(&resolver);
        let a = a.insert("data", superchunk1)?;

        let c = Folder::new(&resolver);
        let c = c.update("a", resolver.save(a)?);
        let c_cid = resolver.save(c)?;

        let commit1 = Commit::new("First commit", c_cid, None, &resolver);
        let commit1_cid = resolver.save(commit1)?;

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, &resolver)?;

        let b = Folder::new(&resolver);
        let b = b.insert("data", superchunk2)?;

        let c = resolver.get_folder(&c_cid)?;
        let c = c.update("b", resolver.save(b)?);
        let c_cid = resolver.save(c)?;

        let commit2 = Commit::new("Second commit", c_cid, Some(commit1_cid), &resolver);

        let cid = resolver.save(commit2)?;

        // Read DAG structure
        let commit = resolver.get_commit(&cid)?;
        assert_eq!(commit.message(), "Second commit");

        let ls = resolver.ls(&cid)?.expect("Couldn't find commit");
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root().size());
        assert_eq!(ls[1].name, String::from("prev"));
        assert_eq!(ls[1].cid, commit.prev.unwrap());
        assert_eq!(ls[1].node_type.unwrap(), "Commit");
        assert_eq!(ls[1].size.unwrap(), commit.prev().unwrap().unwrap().size());

        let ls = resolver.ls(&commit.root)?.expect("Couldn't find folder");
        assert_eq!(ls.len(), 2);

        let c = commit.root();
        let a_cid = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a_cid)?;
        assert_eq!(ls[0].name, String::from("a"));
        assert_eq!(ls[0].cid, a_cid);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), a.size());

        let b_cid = c.get("b").expect("no value for b");
        let b = resolver.get_folder(&b_cid)?;
        assert_eq!(ls[1].name, String::from("b"));
        assert_eq!(ls[1].cid, b_cid);
        assert_eq!(ls[1].node_type.unwrap(), "Folder");
        assert_eq!(ls[1].size.unwrap(), b.size());

        let ls = resolver.ls(&a_cid)?.expect("Couldn't find folder");
        assert_eq!(ls.len(), 1);

        let data_cid = a.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk(&data_cid)?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);
        assert_eq!(ls[0].name, String::from("data"));
        assert_eq!(ls[0].cid, data_cid);
        assert_eq!(ls[0].node_type.unwrap(), "Superchunk");
        assert_eq!(ls[0].size.unwrap(), superchunk.size());

        let superchunk = resolver.get_superchunk(&b.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let cid = commit.prev.unwrap();
        let commit = commit.prev()?.expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let ls = resolver.ls(&cid)?.expect("Couldn't find commit");
        assert_eq!(ls.len(), 1);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root().size());

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a)?;

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev()?.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn make_a_couple_of_commits_async() -> Result<()> {
        // Store DAG structure
        let resolver = testing::resolver();
        let data1 = testing::array(16);
        let superchunk1 = testing::superchunk(&data1, &resolver)?;

        let a = Folder::new(&resolver);
        let a = a.insert_async("data", superchunk1).await?;

        let c = Folder::new(&resolver);
        let c = c.update("a", resolver.save_async(a).await?);
        let c_cid = resolver.save_async(c).await?;

        let commit1 = Commit::new("First commit", c_cid, None, &resolver);
        let commit1_cid = resolver.save_async(commit1).await?;

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, &resolver)?;

        let b = Folder::new(&resolver);
        let b = b.insert_async("data", superchunk2).await?;

        let c = resolver.get_folder_async(&c_cid).await?;
        let c = c.update("b", resolver.save_async(b).await?);
        let c_cid = resolver.save_async(c).await?;

        let commit2 = Commit::new("Second commit", c_cid, Some(commit1_cid), &resolver);

        let cid = resolver.save_async(commit2).await?;

        // Read DAG structure
        let commit = resolver.get_commit_async(&cid).await?;
        assert_eq!(commit.message(), "Second commit");

        let ls = resolver.ls(&cid)?.expect("Couldn't find commit");
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root_async().await.size());
        assert_eq!(ls[1].name, String::from("prev"));
        assert_eq!(ls[1].cid, commit.prev.unwrap());
        assert_eq!(ls[1].node_type.unwrap(), "Commit");
        assert_eq!(
            ls[1].size.unwrap(),
            commit.prev_async().await.unwrap().unwrap().size()
        );

        let ls = resolver.ls(&commit.root)?.expect("Couldn't find folder");
        assert_eq!(ls.len(), 2);

        let c = commit.root_async().await;
        let a_cid = c.get("a").expect("no value for a");
        let a = resolver.get_folder_async(&a_cid).await?;
        assert_eq!(ls[0].name, String::from("a"));
        assert_eq!(ls[0].cid, a_cid);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), a.size());

        let b_cid = c.get("b").expect("no value for b");
        let b = resolver.get_folder_async(&b_cid).await?;
        assert_eq!(ls[1].name, String::from("b"));
        assert_eq!(ls[1].cid, b_cid);
        assert_eq!(ls[1].node_type.unwrap(), "Folder");
        assert_eq!(ls[1].size.unwrap(), b.size());

        let ls = resolver.ls(&a_cid)?.expect("Couldn't find folder");
        assert_eq!(ls.len(), 1);

        let data_cid = a.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk_async(&data_cid).await?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);
        assert_eq!(ls[0].name, String::from("data"));
        assert_eq!(ls[0].cid, data_cid);
        assert_eq!(ls[0].node_type.unwrap(), "Superchunk");
        assert_eq!(ls[0].size.unwrap(), superchunk.size());

        let superchunk = resolver
            .get_superchunk_async(&b.get("data").expect("no value for data"))
            .await?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let cid = commit.prev.unwrap();
        let commit = commit
            .prev_async()
            .await?
            .expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let ls = resolver.ls(&cid)?.expect("Couldn't find commit");
        assert_eq!(ls.len(), 1);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root_async().await.size());

        let c = commit.root_async().await;
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder_async(&a).await?;

        let superchunk = resolver
            .get_superchunk_async(&a.get("data").expect("no value for data"))
            .await?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev_async().await?.is_none());

        Ok(())
    }
}
