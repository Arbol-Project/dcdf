use std::fmt::Debug;
use std::io::Read;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::errors::Result;
use crate::extio::Serialize;

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
    const NODE_TYPE: u8 = NODE_COMMIT; // SMELL Not actually used

    fn store(self, resolver: &Arc<Resolver<N>>) -> Result<Cid> {
        let mut writer = resolver.store();
        writer.write_all(self.message.as_bytes())?;
        let message = writer.finish();

        let mut commit = resolver.init();
        commit = commit.update("root", &self.root);
        if let Some(prev) = self.prev {
            commit = commit.update("prev", &prev);
        }
        commit = commit.update("message.txt", &message);

        Ok(commit.cid())
    }

    fn retrieve(resolver: &Arc<Resolver<N>>, cid: &Cid) -> Result<Option<Self>> {
        let commit = resolver.get_folder(cid);

        let message_txt = commit.get("message.txt").expect("missing message.txt");
        let mut reader = resolver.load(&message_txt.cid).expect("missing object");
        let mut message = String::new();
        reader.read_to_string(&mut message)?;

        let root = commit.get("root").expect("missing root").cid;
        let prev = match commit.get("prev") {
            Some(item) => Some(item.cid),
            None => None,
        };

        Ok(Some(Self {
            root,
            prev,
            message,

            resolver: Arc::clone(resolver),
        }))
    }
}

impl<N> Serialize for Commit<N>
where
    N: Float + Debug + 'static,
{
    // SMELL Poorly factored, must implement but not used
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
