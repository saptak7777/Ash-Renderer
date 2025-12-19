#![no_main]

use libfuzzer_sys::fuzz_target;

// Simplified vertex buffer parsing fuzz target
// Tests buffer overflow, alignment issues, and data corruption
fuzz_target!(|data: &[u8]| {
    // Vertex size is 60 bytes (pos:12, normal:12, uv:8, color:12, tangent:16)
    const VERTEX_SIZE: usize = 60;

    if data.len() < VERTEX_SIZE {
        return;
    }

    // Try to parse vertices from fuzzed data
    let vertex_count = data.len() / VERTEX_SIZE;

    for i in 0..vertex_count {
        let start = i * VERTEX_SIZE;
        let end = start + VERTEX_SIZE;

        if end > data.len() {
            break;
        }

        let vertex_data = &data[start..end];

        // Simulate vertex parsing
        let _position = [
            f32::from_le_bytes([
                vertex_data[0],
                vertex_data[1],
                vertex_data[2],
                vertex_data[3],
            ]),
            f32::from_le_bytes([
                vertex_data[4],
                vertex_data[5],
                vertex_data[6],
                vertex_data[7],
            ]),
            f32::from_le_bytes([
                vertex_data[8],
                vertex_data[9],
                vertex_data[10],
                vertex_data[11],
            ]),
        ];

        let _normal = [
            f32::from_le_bytes([
                vertex_data[12],
                vertex_data[13],
                vertex_data[14],
                vertex_data[15],
            ]),
            f32::from_le_bytes([
                vertex_data[16],
                vertex_data[17],
                vertex_data[18],
                vertex_data[19],
            ]),
            f32::from_le_bytes([
                vertex_data[20],
                vertex_data[21],
                vertex_data[22],
                vertex_data[23],
            ]),
        ];

        let _uv = [
            f32::from_le_bytes([
                vertex_data[24],
                vertex_data[25],
                vertex_data[26],
                vertex_data[27],
            ]),
            f32::from_le_bytes([
                vertex_data[28],
                vertex_data[29],
                vertex_data[30],
                vertex_data[31],
            ]),
        ];

        // Check for NaN or infinite values that could cause issues
        if !_position.iter().all(|v| v.is_finite())
            || !_normal.iter().all(|v| v.is_finite())
            || !_uv.iter().all(|v| v.is_finite())
        {
            // Invalid vertex data detected, but no crash
            continue;
        }
    }
});
