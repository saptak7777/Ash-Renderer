use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders");

    // Create shader compiler
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let shader_dir = Path::new("shaders");

    // We will output .spv files alongside the source files in the shaders directory
    // This allows runtime loading to work as expected by the application

    visit_dirs(shader_dir, &|entry| {
        let path = entry.path();
        if let Some(extension) = path.extension() {
            let extension = extension.to_string_lossy();
            let kind = match extension.as_ref() {
                "vert" => Some(shaderc::ShaderKind::Vertex),
                "frag" => Some(shaderc::ShaderKind::Fragment),
                "comp" => Some(shaderc::ShaderKind::Compute),
                _ => None,
            };

            if let Some(kind) = kind {
                println!("cargo:rerun-if-changed={}", path.display());
                compile_shader(&compiler, &options, &path, kind);
            }
        }
    })
    .expect("Failed to process shader directory");
}

fn visit_dirs(dir: &Path, cb: &dyn Fn(&fs::DirEntry)) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(&entry);
            }
        }
    }
    Ok(())
}

fn compile_shader(
    compiler: &shaderc::Compiler,
    options: &shaderc::CompileOptions,
    path: &Path,
    kind: shaderc::ShaderKind,
) {
    let source = fs::read_to_string(path).expect("Failed to read shader source");
    let file_name = path.file_name().unwrap().to_string_lossy();

    let binary_result =
        compiler.compile_into_spirv(&source, kind, &file_name, "main", Some(options));

    match binary_result {
        Ok(binary) => {
            let mut out_path = PathBuf::from(path);
            // Append .spv extension (e.g. shader.vert -> shader.vert.spv)
            let mut name = out_path.file_name().unwrap().to_os_string();
            name.push(".spv");
            out_path.set_file_name(name);

            fs::write(&out_path, binary.as_binary_u8()).expect("Failed to write SPIR-V binary");
        }
        Err(e) => {
            panic!("Shader compilation failed for {}: {}", path.display(), e);
        }
    }
}
