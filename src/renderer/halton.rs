/// Halton sequence generator for TAA jitter
///
/// Generates a low-discrepancy sequence for sub-pixel camera jitter.
/// This is the "heartbeat" of TAA - without it, TAA is just a blur filter.
pub fn halton_sequence(index: usize, base: usize) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0;
    let mut i = index;

    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }

    result
}

/// Get 2D Halton jitter for TAA (8-sample pattern)
///
/// Returns (x, y) in range [0, 1]. Caller should map to [-0.5, 0.5] pixel offsets.
pub fn halton_jitter_2d(frame_index: usize) -> (f32, f32) {
    let sample = frame_index % 8; // 8-sample pattern
    let x = halton_sequence(sample + 1, 2);
    let y = halton_sequence(sample + 1, 3);
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_sequence() {
        // Base 2 sequence: 0.5, 0.25, 0.75, 0.125, ...
        assert!((halton_sequence(0, 2) - 0.5).abs() < 0.01);
        assert!((halton_sequence(1, 2) - 0.25).abs() < 0.01);
        assert!((halton_sequence(2, 2) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_halton_jitter_2d() {
        let (x, y) = halton_jitter_2d(0);
        assert!(x >= 0.0 && x <= 1.0);
        assert!(y >= 0.0 && y <= 1.0);
    }
}
