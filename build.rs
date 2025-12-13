use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=src/shaders");

    let shader_dir = Path::new("shaders");
    if shader_dir.exists() {
        compile_shaders(shader_dir).expect("Failed to compile shaders");
    }
}

fn compile_shaders(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            compile_shaders(&path)?;
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
                let mut out_path = path.clone();
                // append .spv to the filename
                let new_name = format!("{file_name}.spv");
                out_path.set_file_name(new_name);

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
