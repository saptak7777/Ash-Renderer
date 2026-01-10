use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=src/shaders");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let shader_dir = Path::new("shaders");
    let src_shader_dir = Path::new("src/shaders");

    if shader_dir.exists() {
        compile_shaders(shader_dir, &out_dir).expect("Failed to compile shaders");
    }
    if src_shader_dir.exists() {
        compile_shaders(src_shader_dir, &out_dir).expect("Failed to compile src shaders");
    }
}

fn compile_shaders(dir: &Path, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    // Add DEBUG_VISUALIZATION macro if feature is enabled
    let debug_visualization = env::var("CARGO_FEATURE_DEBUG_VISUALIZATION").is_ok();
    if debug_visualization {
        options.add_macro_definition("DEBUG_VISUALIZATION", Some("1"));
    }

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

        let mut src_content = fs::read_to_string(&path)?;

        // Manual include resolution (pro coder style: robust and compiler-agnostic)
        let mut resolved_content = String::new();
        for line in src_content.lines() {
            if line.trim().starts_with("#include \"") {
                let start = line.find('"').unwrap() + 1;
                let end = line.rfind('"').unwrap();
                let include_name = &line[start..end];
                let include_path = dir.join(include_name);
                if include_path.exists() {
                    let include_src = fs::read_to_string(&include_path)?;
                    resolved_content.push_str(&include_src);
                    resolved_content.push('\n');
                } else {
                    return Err(format!("Include file not found: {:?}", include_path).into());
                }
            } else {
                resolved_content.push_str(line);
                resolved_content.push('\n');
            }
        }
        src_content = resolved_content;

        // Ensure nonuniform_qualifier is enabled for all shaders if they use it
        if src_content.contains("nonuniformEXT")
            && !src_content.contains("GL_EXT_nonuniform_qualifier")
        {
            if let Some(version_end) = src_content.find("\n") {
                let (version, rest) = src_content.split_at(version_end + 1);
                src_content =
                    format!("{version}#extension GL_EXT_nonuniform_qualifier : enable\n{rest}");
            }
        }

        let file_name = path.file_name().unwrap().to_str().unwrap();

        let binary_result =
            compiler.compile_into_spirv(&src_content, kind, file_name, "main", Some(&options));

        match binary_result {
            Ok(binary) => {
                // Use full filename + .spv (e.g., shader.vert.spv)
                let new_name = format!("{file_name}.spv");

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
