use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from("shaders");

    println!("cargo:rerun-if-changed=shaders");

    if !shader_dir.exists() {
        return;
    }

    let entries = fs::read_dir(&shader_dir).expect("Failed to read shader directory");

    for entry in entries {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();

        if let Some(extension) = path.extension() {
            let file_name = path.file_name().unwrap().to_string_lossy();
            let extension_str = extension.to_string_lossy();
            let kind = match extension_str.as_ref() {
                "vert" => shaderc::ShaderKind::Vertex,
                "frag" => shaderc::ShaderKind::Fragment,
                "comp" => shaderc::ShaderKind::Compute,
                "glsl" => {
                    if file_name.contains("vert") {
                        shaderc::ShaderKind::Vertex
                    } else if file_name.contains("frag") {
                        shaderc::ShaderKind::Fragment
                    } else if file_name.contains("comp") {
                        shaderc::ShaderKind::Compute
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            println!("cargo:rerun-if-changed=shaders/{}", file_name);

            let source = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));

            let mut compiler = shaderc::Compiler::new().unwrap();
            let mut options = shaderc::CompileOptions::new().unwrap();
            options.set_target_env(
                shaderc::TargetEnv::Vulkan,
                shaderc::EnvVersion::Vulkan1_2 as u32,
            );
            // options.set_optimization_level(shaderc::OptimizationLevel::Performance);

            let binary_result = compiler
                .compile_into_spirv(&source, kind, &file_name, "main", Some(&options))
                .unwrap_or_else(|e| panic!("Failed to compile {}: {}", path.display(), e));

            let out_path = out_dir.join(format!("{}.spv", file_name));
            fs::write(&out_path, binary_result.as_binary_u8())
                .unwrap_or_else(|e| panic!("Failed to write spirv {}: {}", out_path.display(), e));
        }
    }
}
