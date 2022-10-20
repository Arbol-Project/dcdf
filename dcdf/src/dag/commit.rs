use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite, Serialize};

use super::folder::Folder;
use super::node::{Node, NODE_COMMIT};
use super::resolver::Resolver;

pub struct Commit<N>
where
    N: Float + Debug + 'static,
{
    message: String,
    prev: Option<Cid>,
    root: Cid,

    resolver: Arc<Resolver<N>>,
}

impl<N> Commit<N>
where
    N: Float + Debug + 'static,
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

    pub fn root(&self) -> Arc<Folder<N>> {
        self.resolver.get_folder(&self.root)
    }
}

impl<N> Node<N> for Commit<N>
where
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8 = NODE_COMMIT;

    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        Self::read_header(stream)?;
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
}

impl<N> Serialize for Commit<N>
where
    N: Float + Debug + 'static,
{
    fn write_to(&self, stream: &mut impl io::Write) -> Result<()> {
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

        let a = resolver.init();
        let a = a.insert("data", superchunk1)?;

        let c = resolver.init();
        let c = c.update("a", &a.cid());

        let commit1 = Commit::new("First commit", c.cid(), None, &resolver);
        let commit1_cid = resolver.save(commit1)?;

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, &resolver)?;

        let b = resolver.init();
        let b = b.insert("data", superchunk2)?;

        let c = c.update("b", &b.cid());

        let commit2 = Commit::new("Second commit", c.cid(), Some(commit1_cid), &resolver);

        let cid = resolver.save(commit2)?;

        // Read DAG structure
        let commit = resolver.get_commit(&cid)?;
        assert_eq!(commit.message(), "Second commit");

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a.cid);
        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder(&b.cid);

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data").cid)?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = resolver.get_superchunk(&b.get("data").expect("no value for data").cid)?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let commit = commit.prev()?.expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a.cid);

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data").cid)?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev()?.is_none());

        Ok(())
    }
}
