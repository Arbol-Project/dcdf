use super::*;

mod bitmap {
    use super::*;

    #[test]
    fn new() {
        let bitmap = BitMap::new();

        assert_eq!(bitmap.length, 0);
        assert_eq!(bitmap.packed, vec![]);
    }

    #[test]
    fn push() {
        let mut bitmap = BitMap::new();

        bitmap.push(true);
        assert_eq!(bitmap.length, 1);
        assert_eq!(bitmap.packed, vec![128]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 2);
        assert_eq!(bitmap.packed, vec![128]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 3);
        assert_eq!(bitmap.packed, vec![160]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 4);
        assert_eq!(bitmap.packed, vec![160]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 5);
        assert_eq!(bitmap.packed, vec![168]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 6);
        assert_eq!(bitmap.packed, vec![168]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 7);
        assert_eq!(bitmap.packed, vec![170]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 8);
        assert_eq!(bitmap.packed, vec![170]);

        bitmap.push(false);
        assert_eq!(bitmap.length, 9);
        assert_eq!(bitmap.packed, vec![170, 0]);

        bitmap.push(true);
        assert_eq!(bitmap.length, 10);
        assert_eq!(bitmap.packed, vec![170, 64]);
    }

    #[test]
    fn iter() {
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        let result: Vec<bool> = bitmap.iter().collect();
        assert_eq!(
            result,
            vec![true, false, true, false, true, false, true, false, false, true]
        )
    }

    #[test]
    fn rank1() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        assert_eq!(bitmap.rank1(0), 1);
        assert_eq!(bitmap.rank1(1), 1);
        assert_eq!(bitmap.rank1(2), 2);
        assert_eq!(bitmap.rank1(3), 2);
        assert_eq!(bitmap.rank1(4), 3);
        assert_eq!(bitmap.rank1(5), 3);
        assert_eq!(bitmap.rank1(6), 4);
        assert_eq!(bitmap.rank1(7), 4);
        assert_eq!(bitmap.rank1(8), 4);
        assert_eq!(bitmap.rank1(9), 5);
    }

    #[test]
    #[should_panic]
    fn rank1_out_of_bounds() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        bitmap.rank1(10);
    }

    #[test]
    fn select1() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        assert_eq!(bitmap.select1(1), 0);
        assert_eq!(bitmap.select1(2), 2);
        assert_eq!(bitmap.select1(3), 4);
        assert_eq!(bitmap.select1(4), 6);
        assert_eq!(bitmap.select1(5), 9);
    }

    #[test]
    #[should_panic]
    fn select1_out_of_bounds() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        bitmap.select1(6);
    }

    #[test]
    #[should_panic]
    fn select1_zeroth() {
        // 1010101001
        let bitmap = BitMap {
            length: 10,
            packed: vec![170, 64],
        };

        bitmap.select1(0);
    }
}
