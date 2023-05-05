/// An implementation of an LRU (Least Recently Used) cache.
///
use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::Arc};

use futures::{
    channel::oneshot::{channel, Sender},
    future::BoxFuture,
};

use parking_lot::Mutex;

use crate::errors::{Error, Result};

/// An LRU (least recently used) cache.
///
/// Values must implement ``Cacheable``, which self reports size, intended to be the number of
/// bytes (more or less) an object takes up in memory. For the purposes of DCDF we use the number
/// of bytes in the serialized representation, which will be pretty close to the number of bytes in
/// RAM.
///
/// The ``limit`` is passed in when instantiating the Cache. When an object is added to the cache
/// which causes the total size of objects stored in the cache to exceed the limit, objects are
/// evicted from the cache until the total size is back under the limit. Objects are evicted in
/// least recently used order--objects used longer ago in time chronologically are evicted first.
///
/// The ``Cache`` is designed to be thread safe. When interrogating the cache with ``get``, a
/// ``load`` function is passed in that can be used to load the object from an underlying object
/// store in the event of a cache miss. If multiple requests for the same object come in that
/// require loading the object from the underlying store, the object will only be loaded once, with
/// all requests waiting for the single load to finish before returning to their respective
/// callers.
///
/// The keys are assumed, generally, to be content ids and the values are assumed to be immutable,
/// so nothing special needs to be done for cache invalidation. (If a value changes, so will its
/// key.)
///
pub struct Cache<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    /// The actual cache
    recent: Mutex<Entries<K, V>>,

    /// Synchronization objects for objects that are currently being loaded from the underlying
    /// data store.
    loaders: Mutex<HashMap<K, Arc<Loader<V>>>>,
}

/// A trait for objects that can be cached
///
/// Cacheable objects must be able to self report their size via the ``size`` method.
///
pub trait Cacheable: Sized {
    /// Return the number of bytes in the serialized representation
    fn size(&self) -> u64;
}

/// A structure used to synchronize an object's load operation among all the threads waiting for
/// that object.
struct Loader<V>
where
    V: Cacheable,
{
    /// The loaded entry. If object is not yet loaded, this will be ``None``. Otherwise, it will
    /// contain the loaded object.
    object: Mutex<Option<Result<Arc<V>>>>,

    /// Tasks waiting for this object to be loaded
    waiters: Mutex<Vec<Sender<Result<Arc<V>>>>>,
}

/// A structure containing the entries stored in this cache.
///
/// Entries are directly accessible via ``map`` and also stored in a doubly
/// linked list where ``most_recent`` and ``least_recent`` are the two ends.
///
struct Entries<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    /// Sum of sizes of all entries must stay below this limit. If adding a new entry to the cache
    /// causes this limit to be exceeded, entries are evicted from the cache, starting with the
    /// ``least_recent`` end of the linked list structure.
    limit: u64,

    /// Current sum of sizes of all entries.
    size: u64,

    /// Direct mapping from key to cache entry
    map: HashMap<K, CacheEntry<K, V>>,

    /// The most recently used key
    most_recent: Option<K>,

    /// The least recently used key
    least_recent: Option<K>,
}

/// An entry in the cache
struct CacheEntry<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    /// The key for the object in this entry
    key: K,

    /// The object stored by this entry
    object: Arc<V>,

    /// The next more recent key
    more_recent: Option<K>,

    /// The next less recent key
    less_recent: Option<K>,

    /// The size of this entry, as reported by the object's ``Cacheable::size`` method
    size: u64,
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    /// Instantiate an empty cache with given size limit.
    ///
    pub fn new(limit: u64) -> Self {
        let recent = Mutex::new(Entries {
            limit,
            size: 0,
            map: HashMap::new(),
            most_recent: None,
            least_recent: None,
        });
        let loaders = Mutex::new(HashMap::new());

        Self { recent, loaders }
    }

    /// Get an object by key
    ///
    /// If object isn't in the cache, will call ``load`` to load the object and then store it in
    /// the cache. If the same object is already being loaded in another thread, this will wait for
    /// that object load to finish and then return.
    ///
    pub async fn get<L>(&self, key: &K, load: L) -> Result<Arc<V>>
    where
        L: FnOnce(K) -> BoxFuture<'static, Result<Option<V>>>,
    {
        let object = match self.lookup(key) {
            Some(object) => object,
            None => self.load(key, load).await?,
        };

        Ok(object)
    }

    /// Check if an object is already stored in the cache. If it is, move it to the most recently
    /// used position in the linked list and then return a new reference to it.
    ///
    fn lookup(&self, key: &K) -> Option<Arc<V>> {
        let mut entries = self.recent.lock();
        let entry = entries.remove(key);
        match entry {
            None => None,
            Some(entry) => {
                let object = Arc::clone(&entry.object);
                entries.push_most_recent(entry);
                Some(object)
            }
        }
    }

    /// Load an object from the underlying data store.
    ///
    /// If another thread is already loading the object, wait for that thread.
    ///
    async fn load<L>(&self, key: &K, load: L) -> Result<Arc<V>>
    where
        L: FnOnce(K) -> BoxFuture<'static, Result<Option<V>>>,
    {
        let (first, loader) = {
            let mut loaders = self.loaders.lock();
            match loaders.get(&key) {
                Some(loader) => (false, Arc::clone(loader)),
                None => {
                    let loader = Arc::new(Loader::new());
                    loaders.insert(*key, Arc::clone(&loader));

                    (true, loader)
                }
            }
        };

        if first {
            // We're the first thread to try and load this object, so we'll load it here
            match load(*key).await {
                Ok(result) => {
                    match result {
                        Some(object) => {
                            let object = Arc::new(object);
                            self.recent.lock().insert(*key, &object);

                            // Tell any waiting threads we've finished loading the object
                            loader.finish(Ok(Arc::clone(&object)));
                            self.loaders.lock().remove(key);

                            Ok(object)
                        }
                        None => {
                            // We tried
                            loader.finish(Err(Error::Load));
                            self.loaders.lock().remove(key);
                            panic!();
                        }
                    }
                }
                Err(err) => {
                    // We tried
                    loader.finish(Err(Error::Load));
                    self.loaders.lock().remove(key);
                    Err(err)
                }
            }
        } else {
            // Another thread is loading this object already, just wait for it to finish
            loader.wait().await
        }
    }
}

impl<V> Loader<V>
where
    V: Cacheable,
{
    fn new() -> Self {
        Loader {
            object: Mutex::new(None),
            waiters: Mutex::new(Vec::new()),
        }
    }

    /// Inform any waiting threads that the object has been loaded, or the loading thread has given
    /// up trying.
    ///
    fn finish(&self, object: Result<Arc<V>>) {
        *self.object.lock() = Some(match &object {
            Ok(object) => Ok(Arc::clone(object)),
            Err(_) => Err(Error::Load),
        });
        let mut waiters = self.waiters.lock();
        for waiter in waiters.drain(..) {
            let result = waiter.send(match &object {
                Ok(object) => Ok(Arc::clone(object)),
                Err(_) => Err(Error::Load),
            });
            if let Err(_) = result {
                panic!("Other end hung up!");
            }
        }
    }

    /// Wait for the loading thread to finish loading the object, or give up trying.
    async fn wait(&self) -> Result<Arc<V>> {
        let receive = {
            let mut waiters = self.waiters.lock();
            if let Some(object) = &*self.object.lock() {
                return match object {
                    Ok(object) => Ok(Arc::clone(&object)),
                    Err(_) => Err(Error::Load),
                };
            }
            let (send, receive) = channel::<Result<Arc<V>>>();
            waiters.push(send);

            receive
        };

        receive.await.expect("channel closed unexpectedly")
    }
}

impl<K, V> Entries<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    /// Move an entry to the most recently used spot in the linked list.
    ///
    fn push_most_recent(&mut self, entry: CacheEntry<K, V>) {
        let old_head_key = self.most_recent;

        // Try to short circuit this operation
        if let Some(old_head_key) = old_head_key {
            if old_head_key == entry.key {
                // Already at head, nothing to do
                return;
            }

            // The old head needs to be updated to point to new head in the more recent link
            let old_head = self
                .map
                .remove(&old_head_key)
                .expect("Missing key {old_head_key:?}");
            let less_recent = old_head.less_recent;
            let old_head = old_head.update(Some(entry.key), less_recent);
            self.map.insert(old_head_key, old_head);
        }

        let entry = entry.update(None, old_head_key);
        self.most_recent = Some(entry.key);
        if self.least_recent.is_none() {
            // This is only object in the list, so it is also the tail
            self.least_recent = Some(entry.key);
        }
        self.map.insert(entry.key, entry);
    }

    /// Remove an entry from the cache
    ///
    fn remove(&mut self, key: &K) -> Option<CacheEntry<K, V>> {
        match self.map.remove(key) {
            None => None,
            Some(entry) => {
                if self.most_recent.unwrap() == entry.key {
                    self.most_recent = entry.less_recent;
                }

                if self.least_recent.unwrap() == entry.key {
                    self.least_recent = entry.more_recent;
                }

                if let Some(key) = entry.less_recent {
                    let less_recent = self.map.remove(&key).expect("Missing key {key:?}");
                    let less_recent_less_recent = less_recent.less_recent;
                    let less_recent =
                        less_recent.update(entry.more_recent, less_recent_less_recent);
                    self.map.insert(key, less_recent);
                }

                if let Some(key) = entry.more_recent {
                    let more_recent = self.map.remove(&key).expect("Missing key {key:?}");
                    let more_recent_more_recent = more_recent.more_recent;
                    let more_recent =
                        more_recent.update(more_recent_more_recent, entry.less_recent);
                    self.map.insert(key, more_recent);
                }

                Some(entry)
            }
        }
    }

    /// Add a new object to the cache.
    ///
    /// If the addition of this object causes ``size`` to exceed ``limit``, entries will be evicted
    /// until ``size`` is at or below ``limit`` again before returning.
    ///
    fn insert(&mut self, key: K, object: &Arc<V>) {
        let entry = CacheEntry::new(key, object);
        self.size += entry.size;
        self.push_most_recent(entry);

        // Enforce size limit by removing objects from tail (least recent) until size is within
        // limit
        while self.size > self.limit {
            let evicted = self.remove(&self.least_recent.unwrap()).unwrap();
            self.size -= evicted.size;
        }
    }
}

impl<K, V> CacheEntry<K, V>
where
    K: Eq + Hash + Copy + Debug,
    V: Cacheable,
{
    fn new(key: K, object: &Arc<V>) -> Self {
        Self {
            key,
            object: Arc::clone(object),
            more_recent: None,
            less_recent: None,
            size: object.size(),
        }
    }

    /// Create a copy of this cache entry with updated links to next entries in chain.
    ///
    /// Creating new entries rather than mutating existing ones makes getting along with Rust's
    /// borrow checker much easier. The immutable data structure folks seem to be onto something.
    ///
    fn update(self, more_recent: Option<K>, less_recent: Option<K>) -> Self {
        Self {
            key: self.key,
            object: self.object,
            more_recent,
            less_recent,
            size: self.size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::future::{join_all, FutureExt};
    use std::time::Duration;
    use tokio::time;

    struct DummyValue {
        value: u32,
        size: u64,
    }

    impl DummyValue {
        fn new(value: u32, size: u64) -> Self {
            DummyValue { value, size }
        }
    }

    impl Cacheable for DummyValue {
        fn size(&self) -> u64 {
            self.size
        }
    }

    fn collect_linked_list(recent: &Entries<u32, DummyValue>) -> Vec<u32> {
        // From front to back
        let mut frontwise: Vec<u32> = vec![];
        let mut current = &recent.most_recent;
        loop {
            let next = match *current {
                None => break,
                Some(key) => {
                    let node = recent.map.get(&key).expect("Missing key {key:?}");
                    frontwise.push(node.object.value);
                    &node.less_recent
                }
            };
            current = next;
        }

        // From back to front
        let mut backwise: Vec<u32> = vec![];
        let mut current = &recent.least_recent;
        loop {
            let next = match *current {
                None => break,
                Some(key) => {
                    let node = recent.map.get(&key).expect("Missing key {key:?}");
                    backwise.push(node.object.value);
                    &node.more_recent
                }
            };
            current = next;
        }
        backwise.reverse();

        assert_eq!(frontwise, backwise);

        frontwise
    }

    #[tokio::test]
    async fn test_common_use() -> Result<()> {
        let cache: Cache<u32, DummyValue> = Cache::new(100);
        let load = |key| async move { Ok(Some(DummyValue::new(key, 25))) }.boxed();

        assert_eq!(cache.get(&1, load).await?.value, 1);
        {
            let recent = cache.recent.lock();
            assert_eq!(recent.size, 25);
            assert_eq!(recent.map.len(), 1);
            assert_eq!(collect_linked_list(&recent), vec![1]);
        }

        let load = |_| panic!("I shouldn't get called");
        assert_eq!(cache.get(&1, load).await?.value, 1);

        let load = |key| async move { Ok(Some(DummyValue::new(key, 25))) }.boxed();
        assert_eq!(cache.get(&2, load).await?.value, 2);
        {
            let recent = cache.recent.lock();
            assert_eq!(recent.size, 50);
            assert_eq!(recent.map.len(), 2);
            assert_eq!(collect_linked_list(&recent), vec![2, 1]);
        }

        assert_eq!(cache.get(&3, load).await?.value, 3);
        assert_eq!(cache.get(&4, load).await?.value, 4);
        {
            let recent = cache.recent.lock();
            assert_eq!(recent.size, 100);
            assert_eq!(recent.map.len(), 4);
            assert_eq!(collect_linked_list(&recent), vec![4, 3, 2, 1]);
        }

        let load = |_| panic!("I shouldn't get called");
        assert_eq!(cache.get(&3, load).await?.value, 3);
        {
            let recent = cache.recent.lock();
            assert_eq!(collect_linked_list(&recent), vec![3, 4, 2, 1]);
        }

        assert_eq!(cache.get(&3, load).await?.value, 3);
        {
            let recent = cache.recent.lock();
            assert_eq!(collect_linked_list(&recent), vec![3, 4, 2, 1]);
        }

        assert_eq!(cache.get(&1, load).await?.value, 1);
        {
            let recent = cache.recent.lock();
            assert_eq!(collect_linked_list(&recent), vec![1, 3, 4, 2]);
        }

        // Cache is now full, next load should push 4 and 2 out
        let load = |key| async move { Ok(Some(DummyValue::new(key, 50))) }.boxed();
        assert_eq!(cache.get(&5, load).await?.value, 5);
        {
            let recent = cache.recent.lock();
            assert_eq!(recent.size, 100);
            assert_eq!(recent.map.len(), 3);
            assert_eq!(collect_linked_list(&recent), vec![5, 1, 3]);
        }

        let load = |key| async move { Ok(Some(DummyValue::new(key * 2, 33))) }.boxed();
        assert_eq!(cache.get(&1, load).await?.value, 1);
        assert_eq!(cache.get(&3, load).await?.value, 3);
        assert_eq!(cache.get(&5, load).await?.value, 5);
        assert_eq!(cache.get(&7, load).await?.value, 14);
        assert_eq!(cache.get(&1, load).await?.value, 2);
        assert_eq!(cache.get(&3, load).await?.value, 6);
        assert_eq!(cache.get(&5, load).await?.value, 10);

        // This will obliterate the cache
        let load = |key| async move { Ok(Some(DummyValue::new(key, 101))) }.boxed();
        assert_eq!(cache.get(&7, load).await?.value, 7);
        {
            let recent = cache.recent.lock();
            assert_eq!(recent.size, 0);
            assert_eq!(recent.map.len(), 0);
            assert!(recent.most_recent.is_none());
            assert!(recent.least_recent.is_none());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_load() -> Result<()> {
        let loading = Arc::new(Mutex::new(false));
        let loaded = Arc::new(Mutex::new(false));
        let cache: Arc<Cache<u32, DummyValue>> = Arc::new(Cache::new(100));
        let mut futures = vec![];

        // 10 tasks trying to load single key, only one should actually do the load
        for _ in 1..=10 {
            let loading = Arc::clone(&loading);
            let loaded = Arc::clone(&loaded);
            let cache = Arc::clone(&cache);
            let future = async move {
                let long_load = |key| {
                    async move {
                        // Verify that load is only called once even with many threads trying to load
                        // concurrently.
                        {
                            let mut loading = loading.lock();
                            assert!(!*loading);
                            *loading = true;
                        }

                        // Make all those other threads wait a bit
                        time::sleep(Duration::from_millis(30)).await;

                        // Let the test know we're nearly done loading
                        {
                            let mut loaded = loaded.lock();
                            assert!(!*loaded);
                            *loaded = true;
                        }

                        Ok(Some(DummyValue::new(key, 25)))
                    }
                    .boxed()
                };

                assert_eq!(
                    cache.get(&42, long_load).await.expect("oh no error!").value,
                    42
                );
            }
            .boxed();
            futures.push(future);
        }

        // Another 10 threads loading a different key that shouldn't be blocked by the long load
        // above
        let future = async {
            let mut futures = vec![];
            time::sleep(Duration::from_millis(5)).await;
            for _ in 11..=20 {
                let loaded = Arc::clone(&loaded);
                let loading = Arc::clone(&loading);
                let cache = Arc::clone(&cache);
                let future = async move {
                    let load = |key| {
                        async move {
                            // 5ms wait above should be plenty that the long load has started by now
                            {
                                let loading = loading.lock();
                                assert!(*loading);
                            }

                            Ok(Some(DummyValue::new(key * 2, 25)))
                        }
                        .boxed()
                    };

                    assert_eq!(cache.get(&21, load).await.expect("oh no error!").value, 42);

                    // The long load shouldn't have finished yet
                    {
                        let loaded = loaded.lock();
                        assert!(!*loaded);
                    }
                };
                futures.push(future);
            }

            join_all(futures).await;
        }
        .boxed();

        futures.push(future);
        join_all(futures).await;

        Ok(())
    }
}
