use std::collections::{btree_map, BTreeMap};
use std::fmt::Debug;
use std::sync::Arc;

use cid::Cid;
use num_traits::Float;

use crate::errors::Result;

use super::mapper::Link;
use super::node::Node;
use super::resolver::Resolver;

pub struct Folder<N>
where
    N: Float + Debug + 'static,
{
    cid: Cid,
    items: BTreeMap<String, FolderItem>,
    resolver: Arc<Resolver<N>>,
}

#[derive(Clone)]
pub struct FolderItem {
    pub cid: Cid,
    pub size: u64,
}

impl<N> Folder<N>
where
    N: Float + Debug + 'static,
{
    pub(crate) fn new(resolver: &Arc<Resolver<N>>, cid: Cid, links: Vec<Link>) -> Arc<Self> {
        let mut items = BTreeMap::new();
        for link in links {
            let item = FolderItem {
                cid: link.cid,
                size: link.size,
            };
            items.insert(link.name, item);
        }
        Arc::new(Self {
            cid,
            items,
            resolver: Arc::clone(resolver),
        })
    }

    pub fn update<S>(&self, name: S, object: &Cid) -> Arc<Self>
    where
        S: Into<String>,
    {
        let name = name.into();
        let new_root = self.resolver.insert(&self.cid, &name, object);
        self.resolver.get_folder(&new_root)
    }

    pub fn insert<O, S>(&self, name: S, object: O) -> Result<Arc<Self>>
    where
        O: Node<N>,
        S: Into<String>,
    {
        let name = name.into();
        let object = self.resolver.save(object)?;
        let new_root = self.resolver.insert(&self.cid, &name, &object);

        Ok(self.resolver.get_folder(&new_root))
    }

    pub fn iter(&self) -> btree_map::Iter<String, FolderItem> {
        self.items.iter()
    }

    pub fn get<S>(&self, key: S) -> Option<FolderItem>
    where
        S: Into<String>,
    {
        let key = key.into();
        match self.items.get(&key) {
            Some(value) => Some(value.clone()),
            None => None,
        }
    }

    pub fn cid(&self) -> Cid {
        self.cid
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

        let a = resolver.init();
        let a = a.update("data", &superchunk1);

        let data2 = testing::array(15);
        let superchunk2 = testing::superchunk(&data2, resolver)?;
        let superchunk2 = resolver.save(superchunk2)?;

        let b = resolver.init();
        let b = b.update("data", &superchunk2);

        let c = resolver.init();
        let c = c.update("a", &a.cid);
        let c = c.update("b", &b.cid);
        let c = c.update("d/e/f", &superchunk1);

        Ok(c.cid)
    }

    #[test]
    fn test_iter() -> Result<()> {
        let resolver = testing::resolver();
        let cid = folders(&resolver)?;
        let c = resolver.get_folder(&cid);

        let mut contents = c.iter();
        let (key, item) = contents.next().expect("Expecting folder a");
        assert_eq!(key, "a");
        let a = resolver.get_folder(&item.cid);

        let (key, item) = contents.next().expect("Expecting folder b");
        assert_eq!(key, "b");
        let b = resolver.get_folder(&item.cid);

        let (key, item) = contents.next().expect("Expecting folder d");
        assert_eq!(key, "d");
        let d = resolver.get_folder(&item.cid);

        let (key, item) = d.iter().next().expect("Expecting folder e");
        assert_eq!(key, "e");
        let e = resolver.get_folder(&item.cid);

        let (key, item) = e.iter().next().expect("Expecting superchunk f");
        assert_eq!(key, "f");
        let f = resolver.get_superchunk(&item.cid)?;
        assert_eq!(f.shape(), [100, 16, 16]);

        assert!(contents.next().is_none());

        let mut contents = a.iter();
        let (key, item) = contents.next().expect("Expecting data superchunk");
        assert_eq!(key, "data");
        let superchunk = resolver.get_superchunk(&item.cid)?;

        assert_eq!(superchunk.shape(), [100, 16, 16]);
        assert!(contents.next().is_none());

        let mut contents = b.iter();
        let (key, item) = contents.next().expect("Expecting data superchunk");
        assert_eq!(key, "data");
        let superchunk = resolver.get_superchunk(&item.cid)?;

        assert_eq!(superchunk.shape(), [100, 15, 15]);
        assert!(contents.next().is_none());

        Ok(())
    }

    #[test]
    fn test_get() -> Result<()> {
        let resolver = testing::resolver();
        let cid = folders(&resolver)?;
        let c = resolver.get_folder(&cid);

        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a.cid);

        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder(&b.cid);

        let superchunk = a.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk(&superchunk.cid)?;

        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = b.get("data").expect("no value for data");
        let superchunk = resolver.get_superchunk(&superchunk.cid)?;

        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let d = c.get("d").expect("no value for d");
        let d = resolver.get_folder(&d.cid);

        let e = d.get("e").expect("no value for e");
        let e = resolver.get_folder(&e.cid);

        let f = e.get("f").expect("no value for f");
        let f = resolver.get_superchunk(&f.cid)?;

        assert_eq!(f.shape(), [100, 16, 16]);

        Ok(())
    }
}
