use super::helpers::rearrange;

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
    _private: (),
}

impl Rect {
    pub fn new(top: usize, bottom: usize, left: usize, right: usize) -> Self {
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        Self {
            top,
            bottom,
            left,
            right,
            _private: (),
        }
    }

    pub fn rows(&self) -> usize {
        self.bottom - self.top
    }

    pub fn cols(&self) -> usize {
        self.right - self.left
    }

    pub fn iter(&self) -> RectIter {
        RectIter {
            row: self.top,
            col: self.left,
            left: self.left,
            bottom: self.bottom,
            right: self.right,
        }
    }
}

pub struct RectIter {
    row: usize,
    col: usize,
    left: usize,
    bottom: usize,
    right: usize,
}

impl Iterator for RectIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.bottom {
            let coord = (self.row, self.col);
            self.col += 1;
            if self.col == self.right {
                self.col = self.left;
                self.row += 1;
            }

            Some(coord)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cube {
    pub start: usize,
    pub end: usize,
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
    _private: (),
}

impl Cube {
    pub fn new(
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        Self {
            start,
            end,
            top,
            bottom,
            left,
            right,
            _private: (),
        }
    }

    pub fn instants(&self) -> usize {
        self.end - self.start
    }

    pub fn rows(&self) -> usize {
        self.bottom - self.top
    }

    pub fn cols(&self) -> usize {
        self.right - self.left
    }

    pub fn rect(&self) -> Rect {
        Rect::new(self.top, self.bottom, self.left, self.right)
    }
}
