use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Tell Cargo to re-run this script if shaders change
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = env::var("OUT_DIR").unwrap();
    let compiler = shaderc::Compiler::new().unwrap();

    let shader_dir = Path::new("shaders");
    if !shader_dir.exists() {
        return; // Skip if no shaders (e.g. published crate without sources, though we included them)
    }

    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if let Some(filename_os) = path.file_name() {
            let file_name = filename_os.to_str().unwrap();

            // Handle extensions
            let (kind, output_name) = if file_name.ends_with(".vert") {
                (shaderc::ShaderKind::Vertex, format!("{file_name}.spv"))
            } else if file_name.ends_with(".frag") {
                (shaderc::ShaderKind::Fragment, format!("{file_name}.spv"))
            } else if file_name.ends_with(".comp") {
                (shaderc::ShaderKind::Compute, format!("{file_name}.spv"))
            } else if file_name.ends_with(".glsl") {
                if file_name.contains("vert") {
                    (
                        shaderc::ShaderKind::Vertex,
                        file_name.replace(".glsl", ".spv"),
                    )
                } else if file_name.contains("frag") {
                    (
                        shaderc::ShaderKind::Fragment,
                        file_name.replace(".glsl", ".spv"),
                    )
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let source = fs::read_to_string(&path).unwrap();

            // Compile
            let binary_result = compiler.compile_into_spirv(&source, kind, file_name, "main", None);

            match binary_result {
                Ok(binary) => {
                    let out_path = PathBuf::from(&out_dir).join(&output_name);
                    fs::write(&out_path, binary.as_binary_u8()).unwrap();
                }
                Err(e) => {
                    // Panic to fail build if shader error
                    panic!("Failed to compile shader {file_name}: {e}");
                }
            }
        }
    }
}
