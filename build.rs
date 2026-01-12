// Build script to compile shaders and bake IBL assets
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/textures");
    println!("cargo:rerun-if-changed=shaders");

    // Compile shaders
    compile_shaders();

    // Bake IBL assets
    bake_ibl_assets();
}

fn compile_shaders() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Define shaders to compile: (input_path, output_name, kind, defines)
    let shaders = [
        // Vertex shaders
        ("shaders/vert.vert", "vert.vert.spv", "vert", &[] as &[&str]),
        (
            "shaders/postprocess.vert",
            "postprocess.vert.spv",
            "vert",
            &[],
        ),
        ("shaders/shadow.vert", "shadow.vert.spv", "vert", &[]),
        ("shaders/overlay.vert", "overlay.vert.spv", "vert", &[]),
        ("shaders/triangle.vert", "triangle.vert.spv", "vert", &[]),
        ("shaders/skinning.vert", "skinning.vert.spv", "vert", &[]),
        ("shaders/motion.vert", "motion.vert.spv", "vert", &[]),
        // Fragment shaders
        (
            "shaders/frag.frag",
            "frag.frag.spv",
            "frag",
            &["ENABLE_POINT_LIGHTS"],
        ),
        (
            "shaders/tonemapping.frag",
            "tonemapping.frag.spv",
            "frag",
            &[],
        ),
        ("shaders/shadow.frag", "shadow.frag.spv", "frag", &[]),
        ("shaders/overlay.frag", "overlay.frag.spv", "frag", &[]),
        ("shaders/triangle.frag", "triangle.frag.spv", "frag", &[]),
        ("shaders/brdf_lut.frag", "brdf_lut.frag.spv", "frag", &[]),
        (
            "shaders/bloom_threshold.frag",
            "bloom_threshold.frag.spv",
            "frag",
            &[],
        ),
        (
            "shaders/bloom_prefilter.frag",
            "bloom_prefilter.frag.spv",
            "frag",
            &[],
        ),
        (
            "shaders/bloom_downsample.frag",
            "bloom_downsample.frag.spv",
            "frag",
            &[],
        ),
        (
            "shaders/bloom_upsample.frag",
            "bloom_upsample.frag.spv",
            "frag",
            &[],
        ),
        ("shaders/motion.frag", "motion.frag.spv", "frag", &[]),
        // Compute shaders
        (
            "shaders/light_culling.comp",
            "light_culling.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/taa_resolve.comp",
            "taa_resolve.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/tsr_upscale.comp",
            "tsr_upscale.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/vsr_upscale.comp",
            "vsr_upscale.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/occlusion_cull.comp",
            "occlusion_cull.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/hiz_generate.comp",
            "hiz_generate.comp.spv",
            "comp",
            &[],
        ),
        ("shaders/ssgi.comp", "ssgi.comp.spv", "comp", &[]),
        (
            "shaders/atrous_denoise.comp",
            "atrous_denoise.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/cluster_cull.comp",
            "cluster_cull.comp.spv",
            "comp",
            &[],
        ),
        // IBL shaders
        (
            "shaders/ibl/equirect_to_cubemap.comp",
            "equirect_to_cubemap.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/ibl/irradiance_convolution.comp",
            "irradiance_convolution.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/ibl/prefilter_envmap.comp",
            "prefilter_envmap.comp.spv",
            "comp",
            &[],
        ),
    ];

    for (input, output, kind, defines) in shaders {
        let input_path = Path::new(input);
        let output_path = out_dir.join(output);

        // Check if recompilation is needed
        let needs_recompile = !output_path.exists() || {
            let input_time = fs::metadata(input_path).and_then(|m| m.modified()).ok();
            let output_time = fs::metadata(&output_path).and_then(|m| m.modified()).ok();

            match (input_time, output_time) {
                (Some(i), Some(o)) => i > o,
                _ => true,
            }
        };

        if !needs_recompile {
            continue;
        }

        // Use glslc for compilation (supports #include directives)
        let glslc = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        let mut cmd = Command::new(glslc);
        cmd.arg(input)
            .arg("-o")
            .arg(&output_path)
            .arg(format!("-fshader-stage={kind}"));

        // Add defines
        for define in defines {
            cmd.arg(format!("-D{define}"));
        }

        let status = cmd.status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!(
                    "cargo:warning=Failed to compile shader {input}: exit code {:?}",
                    s.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=Failed to run glslc for {input}: {e}");
            }
        }
    }
}

fn bake_ibl_assets() {
    // Check if ibl_baker binary exists
    let baker_path = if cfg!(windows) {
        "target/debug/ibl_baker.exe"
    } else {
        "target/debug/ibl_baker"
    };

    // Only bake if the baker tool exists (avoid build failures on first compile)
    if !Path::new(baker_path).exists() {
        println!("cargo:warning=ibl_baker not found, skipping asset baking");
        println!(
            "cargo:warning=Run 'cargo build -p ibl_baker' first to enable automatic asset baking"
        );
        return;
    }

    // List of HDR files to bake
    let hdr_files = [("assets/textures/skybox.hdr", "assets/textures/skybox.ibl")];

    for (input, output) in &hdr_files {
        let input_path = Path::new(input);
        let output_path = Path::new(output);

        // Skip if input doesn't exist
        if !input_path.exists() {
            println!("cargo:warning=HDR file not found: {input}");
            continue;
        }

        // Check if we need to rebuild (output missing or input newer)
        let needs_rebuild = !output_path.exists() || {
            let input_time = std::fs::metadata(input_path)
                .and_then(|m| m.modified())
                .ok();
            let output_time = std::fs::metadata(output_path)
                .and_then(|m| m.modified())
                .ok();

            match (input_time, output_time) {
                (Some(i), Some(o)) => i > o,
                _ => true,
            }
        };

        if needs_rebuild {
            println!("cargo:warning=Baking IBL asset: {input} -> {output}");

            let status = Command::new(baker_path)
                .args(["--input", input, "--output", output])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("cargo:warning=Successfully baked {output}");
                }
                Ok(s) => {
                    println!(
                        "cargo:warning=Failed to bake {} (exit code: {:?})",
                        output,
                        s.code()
                    );
                }
                Err(e) => {
                    println!("cargo:warning=Failed to run ibl_baker: {e}");
                }
            }
        }
    }
}
