use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::errors::Result;
use crate::extio::{ExtendedRead, ExtendedWrite};

use super::folder::Folder;
use super::node::{Node, NODE_COMMIT};
use super::resolver::Resolver;

pub struct Commit<N>
where
    N: Float + Debug + 'static,
{
    message: String,
    pub prev: Option<Cid>,
    pub root: Cid,

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
        self.resolver
            .get_folder(&self.root)
            .expect("Root not found")
    }
}

impl<N> Node<N> for Commit<N>
where
    N: Float + Debug + 'static,
{
    const NODE_TYPE: u8 = NODE_COMMIT;
    const NODE_TYPE_STR: &'static str = "Commit";

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

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a)?;
        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder(&b)?;

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = resolver.get_superchunk(&b.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let commit = commit.prev()?.expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a)?;

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev()?.is_none());

        Ok(())
    }
}
