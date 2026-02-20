use glam::Vec2;

#[derive(Debug)]
pub struct HaltonSequence {
    index: u32,
    base_x: u32,
    base_y: u32,
}

impl HaltonSequence {
    pub fn new(base_x: u32, base_y: u32) -> Self {
        assert!(base_x > 1 && base_y > 1, "Halton bases must be > 1");
        Self {
            index: 0,
            base_x,
            base_y,
        }
    }

    fn halton(mut index: u32, base: u32) -> f32 {
        assert!(base > 1, "Halton base must be > 1 to avoid infinite loops");
        let mut f = 1.0;
        let mut r = 0.0;
        while index > 0 {
            f /= base as f32;
            r += f * (index % base) as f32;
            index /= base;
        }
        r
    }

    pub fn next_sample(&mut self) -> Vec2 {
        self.index += 1;
        // Directly map [0, 1] to centered [-0.5, 0.5] range for projection matrices
        Vec2::new(
            Self::halton(self.index, self.base_x) - 0.5,
            Self::halton(self.index, self.base_y) - 0.5,
        )
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_range() {
        let mut seq = HaltonSequence::new(2, 3);
        for _ in 0..100 {
            let sample = seq.next_sample();
            // Ensure all jitter stays within a 1-pixel footprint
            assert!(sample.x >= -0.5 && sample.x <= 0.5);
            assert!(sample.y >= -0.5 && sample.y <= 0.5);
        }
    }
}
