use super::*;

mod bitmap {
    use super::*;

    #[test]
    fn new() {
        let bitmap = BitMap::new();

        assert_eq!(bitmap.length, 0);
        assert_eq!(bitmap.bitmap, vec![]);
    }

    #[test]
    fn push() {
        let mut bitmap = BitMap::new();

        bitmap.push(true);
        assert_eq!(bitmap.length, 1);
        assert_eq!(bitmap.bitmap, vec![128]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 2);
        assert_eq!(bitmap.bitmap, vec![128]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 3);
        assert_eq!(bitmap.bitmap, vec![160]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 4);
        assert_eq!(bitmap.bitmap, vec![160]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 5);
        assert_eq!(bitmap.bitmap, vec![168]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 6);
        assert_eq!(bitmap.bitmap, vec![168]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 7);
        assert_eq!(bitmap.bitmap, vec![170]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 8);
        assert_eq!(bitmap.bitmap, vec![170]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 9);
        assert_eq!(bitmap.bitmap, vec![170, 0]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 10);
        assert_eq!(bitmap.bitmap, vec![170, 64]);
    }

    #[test]
    fn rank() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        assert_eq!(bitmap.rank(0), 0);
        assert_eq!(bitmap.rank(1), 1);
        assert_eq!(bitmap.rank(2), 1);
        assert_eq!(bitmap.rank(3), 2);
        assert_eq!(bitmap.rank(4), 2);
        assert_eq!(bitmap.rank(5), 3);
        assert_eq!(bitmap.rank(6), 3);
        assert_eq!(bitmap.rank(7), 4);
        assert_eq!(bitmap.rank(8), 4);
        assert_eq!(bitmap.rank(9), 4);
        assert_eq!(bitmap.rank(10), 5);
    }

    #[test]
    #[should_panic]
    fn rank_out_of_bounds() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        bitmap.rank(11);
    }

    #[test]
    fn select() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        assert_eq!(bitmap.select(1).unwrap(), 1);
        assert_eq!(bitmap.select(2).unwrap(), 3);
        assert_eq!(bitmap.select(3).unwrap(), 5);
        assert_eq!(bitmap.select(4).unwrap(), 7);
        assert_eq!(bitmap.select(5).unwrap(), 10);
        assert_eq!(bitmap.select(6), None);
    }

    #[test]
    #[should_panic]
    fn select_zeroth() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        bitmap.select(0);
    }
}

mod indexed_bitmap {
    use super::*;
    use rand;
    use rand::{Rng, RngCore};
    use std::time;

    #[test]
    fn from_bitmap() {
        let bitmap = BitMap {
            length: 36,
            bitmap: vec![99, 104, 114, 105, 115],
        };

        let bitmap = IndexedBitMap::from(bitmap);
        assert_eq!(bitmap.bitmap, vec![1667789417, 1929379840]);

        let bitmap = BitMap {
            length: 129,
            bitmap: vec![
                99, 104, 114, 105, 115, 0, 0, 0, 99, 104, 114, 105, 115, 0, 0, 0, 128,
            ],
        };

        let bitmap = IndexedBitMap::from(bitmap);
        assert_eq!(
            bitmap.bitmap,
            vec![1667789417, 1929379840, 1667789417, 1929379840, 1 << 31]
        );
        assert_eq!(bitmap.index, vec![40]);
    }

    fn test_rank(bitmap: BitMap, indexes: &[usize]) {
        // Gather answers using the naive, reference implementation
        println!(
            "Test rank: {} bits, {}/{} lookups",
            bitmap.length,
            indexes.len(),
            indexes.len() * 1000,
        );

        let timer = time::Instant::now();
        let mut answers: Vec<usize> = Vec::with_capacity(indexes.len());
        for index in indexes {
            answers.push(bitmap.rank(*index));
        }
        let reference_impl = timer.elapsed().as_millis();

        // Compare our answers with the reference implementation
        let timer = time::Instant::now();
        let bitmap = IndexedBitMap::from(bitmap);
        let make_index = timer.elapsed().as_millis();

        let timer = time::Instant::now();
        for _ in 1..1000 {
            for (index, answer) in indexes.iter().zip(answers.iter()) {
                assert_eq!(*answer, bitmap.rank(*index));
            }
        }
        let our_impl = timer.elapsed().as_millis();

        println!("time to build index: {}", make_index);
        println!("reference impl: {}, our impl: {}", reference_impl, our_impl);
    }

    #[test]
    fn rank() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        let indexes: Vec<usize> = (0..10).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[should_panic]
    fn rank_out_of_bounds() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        let bitmap = IndexedBitMap::from(bitmap);
        bitmap.rank(11);
    }

    struct RandomRange(usize);

    impl Iterator for RandomRange {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            Some(rand::thread_rng().gen_range(0..self.0))
        }
    }

    fn make_bitmap(n: usize) -> BitMap {
        let timer = time::Instant::now();
        let mut bytes: Vec<u8> = Vec::with_capacity(n >> 3);
        bytes.resize(n >> 3, 0);
        rand::thread_rng().fill_bytes(&mut bytes);

        let bitmap = BitMap {
            length: n,
            bitmap: bytes,
        };
        println!("Time to build bitmap: {}", timer.elapsed().as_millis());

        bitmap
    }

    #[test]
    fn rank_megabit() {
        let bitmap = make_bitmap(1 << 20);
        let indexes = RandomRange(1 << 20);
        let indexes: Vec<usize> = indexes.take(100).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[ignore]
    fn rank_gigabit() {
        let bitmap = make_bitmap(1 << 30);
        let indexes: Vec<usize> = RandomRange(1 << 30).take(1000).collect();
        test_rank(bitmap, &indexes);
    }

    fn test_select(bitmap: BitMap, counts: &[usize]) {
        // Gather answers using the naive, reference implementation
        println!(
            "Test select: {} bits, {}/{} lookups",
            bitmap.length,
            counts.len(),
            counts.len() * 1000,
        );

        let timer = time::Instant::now();
        let mut answers: Vec<Option<usize>> = Vec::with_capacity(counts.len());
        for count in counts {
            answers.push(bitmap.select(*count));
        }
        let reference_impl = timer.elapsed().as_millis();

        // Compare our answers with the reference implementation
        let timer = time::Instant::now();
        let bitmap = IndexedBitMap::from(bitmap);
        let make_index = timer.elapsed().as_millis();

        let timer = time::Instant::now();
        for _ in 0..1000 {
            for (count, answer) in counts.iter().zip(answers.iter()) {
                assert_eq!(*answer, bitmap.select(*count));
            }
        }
        let our_impl = timer.elapsed().as_millis();

        println!("time to build index: {}", make_index);
        println!("reference impl: {}, our impl: {}", reference_impl, our_impl);
    }

    #[test]
    fn select() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };
        test_select(bitmap, &vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn select_zeroth() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            bitmap: vec![170, 64],
        };

        let bitmap = IndexedBitMap::from(bitmap);
        bitmap.select(0);
    }

    #[test]
    fn select_first_at_end_of_block() {
        let mut bitmap = make_bitmap(1 << 20);

        // Contrivance to exercise select_first_at_end_of_block
        // Go to the end of the 200th block and zero out the last word of that block
        let addr = 200 * 4 * 4; // 4 words per block, 4 bytes per word
        for i in addr - 5..addr {
            bitmap.bitmap[i] = 0;
        }
        let rank_200 = bitmap.rank(200 * 4 * 4 * 8);
        test_select(bitmap, &vec![rank_200]);
    }

    #[test]
    fn select_megabit() {
        let bitmap = make_bitmap(1 << 20);
        let indexes: Vec<usize> = RandomRange(1 << 20).take(100).collect();
        test_select(bitmap, &indexes);
    }

    #[test]
    #[ignore]
    fn select_gigabit() {
        let bitmap = make_bitmap(1 << 30);
        let indexes: Vec<usize> = RandomRange(1 << 30).take(1000).collect();
        test_select(bitmap, &indexes);
    }
}
