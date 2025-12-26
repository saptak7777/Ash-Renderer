use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=src/shaders");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let shader_dir = Path::new("shaders");

    if shader_dir.exists() {
        compile_shaders(shader_dir, &out_dir).expect("Failed to compile shaders");
    }
}

fn compile_shaders(dir: &Path, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            compile_shaders(&path, out_dir)?;
            continue;
        }

        let extension = match path.extension().and_then(|s| s.to_str()) {
            Some(ext) => ext,
            None => continue,
        };

        let kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            _ => continue,
        };

        // Skip if it's already an spv file
        if extension == "spv" {
            continue;
        }

        let src_content = fs::read_to_string(&path)?;
        let file_name = path.file_name().unwrap().to_str().unwrap();

        let binary_result =
            compiler.compile_into_spirv(&src_content, kind, file_name, "main", Some(&options));

        match binary_result {
            Ok(binary) => {
                // Special case for backward compatibility (v0.2.7 style)
                let new_name = if file_name == "vert.vert" {
                    "vert.spv".to_string()
                } else if file_name == "frag.frag" {
                    "frag.spv".to_string()
                } else {
                    format!("{file_name}.spv")
                };

                let out_path = out_dir.join(new_name);
                fs::write(&out_path, binary.as_binary_u8())?;
            }
            Err(e) => {
                eprintln!("Failed to compile shader {}: {}", path.display(), e);
                return Err(Box::new(e));
            }
        }
    }
    Ok(())
}
