use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut file = std::fs::File::create("assets/textures/skybox.hdr")?;
    file.write_all(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n")?;
    // White pixel (1.0, 1.0, 1.0) in RGBE is (128, 128, 128, 129 + 1) -> roughly (128, 128, 128, 130)
    // Actually simplicity: RGBE (128, 128, 128, 129) is (0.5, 0.5, 0.5)
    // RGBE (128, 128, 128, 130) is (1.0, 1.0, 1.0)
    file.write_all(&[128, 128, 128, 130])?;
    file.write_all(&[128, 128, 128, 130])?;
    Ok(())
}
