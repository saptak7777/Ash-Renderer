// Build script to compile shaders
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/textures");
    println!("cargo:rerun-if-changed=shaders");

    // Compile shaders
    compile_shaders();
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
        ("shaders/sharpen.comp", "sharpen.comp.spv", "comp", &[]),
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
        (
            "shaders/cluster_cull.comp",
            "cluster_cull.comp.spv",
            "comp",
            &[],
        ),
        (
            "shaders/shadow_cull.comp",
            "shadow_cull.comp.spv",
            "comp",
            &[],
        ),
        ("shaders/skybox.vert", "skybox.vert.spv", "vert", &[]),
        ("shaders/skybox.frag", "skybox.frag.spv", "frag", &[]),
    ];

    for (input, output, kind, defines) in shaders {
        let input_path = Path::new(input);

        // Skip if input doesn't exist (prevents build failure of unneeded shaders)
        if !input_path.exists() {
            continue;
        }

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

        let output = cmd.output().expect("Failed to execute glslc");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "SHADER COMPILATION FAILED:\nFile: {}\nExit Code: {:?}\nError: {}",
                input,
                output.status.code(),
                stderr
            );
        }
    }
}
