// Build script to compile shaders recursively from src/shaders
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

fn main() {
    println!("cargo:rerun-if-changed=assets/textures");
    println!("cargo:rerun-if-changed=src/shaders");

    // Compile shaders
    compile_shaders();
}

fn compile_shaders() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_src_dir = Path::new("src/shaders");

    // Gather all shader files recursively
    for entry in WalkDir::new(shader_src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        let kind = match extension {
            "vert" => "vert",
            "frag" => "frag",
            "comp" => "comp",
            "glsl" if path.to_str().unwrap().contains("compute") => "comp",
            _ => continue, // Skip other files
        };

        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap();
        let out_name = format!("{file_name}.spv");
        let output_path = out_dir.join(&out_name);

        // Check if recompilation is needed
        // Centralized ABI dependency tracking: If any common/interop GLSL file is newer,
        // we must recompile ALL shaders to ensure ABI synchronization.
        let max_include_time = get_max_include_time(shader_src_dir);
        let input_time = fs::metadata(path).and_then(|m| m.modified()).ok();
        let output_time = fs::metadata(&output_path).and_then(|m| m.modified()).ok();

        let needs_recompile = !output_path.exists() || {
            match (input_time, output_time, max_include_time) {
                (Some(i), Some(o), Some(m)) => i > o || m > o,
                (Some(i), Some(o), None) => i > o,
                _ => true,
            }
        };

        if !needs_recompile {
            continue;
        }

        // Use glslc for compilation (supports #include directives)
        let glslc = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        let mut cmd = Command::new(glslc);

        // Add include path for centralized structures
        cmd.arg("-I").arg(shader_src_dir);

        cmd.arg(format!("-fshader-stage={kind}"))
            .arg("-o")
            .arg(&output_path)
            .arg(path);

        // Add special defines
        if file_name == "forward.frag" {
            cmd.arg("-DENABLE_POINT_LIGHTS");
        }

        let output = cmd.output().expect("Failed to execute glslc");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "SHADER COMPILATION FAILED:\nFile: {}\nExit Code: {:?}\nError: {stderr}",
                path.display(),
                output.status.code(),
            );
        }
    }
}

/// Helper to get the maximum modification time among all files in the interop and common shader directories.
/// This ensures that changes to shared ABI structures trigger a full recompile of all dependent shaders.
fn get_max_include_time(shader_src_dir: &Path) -> Option<std::time::SystemTime> {
    let mut max_time: Option<std::time::SystemTime> = None;
    let include_dirs = ["interop", "common"];

    for dir_name in include_dirs {
        let dir_path = shader_src_dir.join(dir_name);
        if !dir_path.exists() {
            continue;
        }

        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Ok(metadata) = fs::metadata(entry.path()) {
                if let Ok(modified) = metadata.modified() {
                    max_time = Some(match max_time {
                        Some(current_max) => current_max.max(modified),
                        None => modified,
                    });
                }
            }
        }
    }

    max_time
}
