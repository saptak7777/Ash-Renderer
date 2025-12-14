use ash::vk;
#[cfg(feature = "shader_reflection")]
use spirv_reflect::ShaderModule as ReflectShaderModule;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use crate::{AshError, Result};

/// Reflection metadata extracted from a SPIR-V shader.
pub struct ShaderReflection {
    pub push_constants: Vec<vk::PushConstantRange>,
    pub descriptor_sets: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'static>>>,
    pub input_attributes: Vec<vk::VertexInputAttributeDescription>,
    pub output_attachment_formats: Vec<vk::Format>,
    pub stage: vk::ShaderStageFlags,
}

impl Default for ShaderReflection {
    fn default() -> Self {
        Self {
            push_constants: Vec::new(),
            descriptor_sets: HashMap::new(),
            input_attributes: Vec::new(),
            output_attachment_formats: Vec::new(),
            stage: vk::ShaderStageFlags::empty(),
        }
    }
}

impl ShaderReflection {
    /// Reflect shader resources from SPIR-V bytecode.
    ///
    /// Requires the `shader_reflection` feature to be enabled.
    #[cfg(feature = "shader_reflection")]
    pub fn reflect(code: &[u8], stage: vk::ShaderStageFlags) -> Result<Self> {
        let module = ReflectShaderModule::load_u8_data(code)
            .map_err(|e| AshError::VulkanError(format!("SPIR-V reflection failed: {e}")))?;

        let mut reflection = ShaderReflection {
            stage,
            ..Default::default()
        };

        let stage_name = match stage {
            vk::ShaderStageFlags::VERTEX => "VERTEX",
            vk::ShaderStageFlags::FRAGMENT => "FRAGMENT",
            vk::ShaderStageFlags::COMPUTE => "COMPUTE",
            vk::ShaderStageFlags::GEOMETRY => "GEOMETRY",
            vk::ShaderStageFlags::TESSELLATION_CONTROL => "TESS_CTRL",
            vk::ShaderStageFlags::TESSELLATION_EVALUATION => "TESS_EVAL",
            _ => "UNKNOWN",
        };

        // Extract push constants
        if let Ok(push_constants) = module.enumerate_push_constant_blocks(None) {
            for block in push_constants {
                log::debug!(
                    "[Shader Reflection] {} push constant: offset={}, size={} bytes",
                    stage_name,
                    block.offset,
                    block.size
                );
                reflection.push_constants.push(vk::PushConstantRange {
                    stage_flags: stage,
                    offset: block.offset,
                    size: block.size,
                });
            }
        }

        // Extract descriptor sets and bindings
        if let Ok(descriptor_sets) = module.enumerate_descriptor_sets(None) {
            for set in descriptor_sets {
                log::debug!(
                    "[Shader Reflection] {} descriptor set {}: {} bindings",
                    stage_name,
                    set.set,
                    set.bindings.len()
                );

                let bindings: Vec<_> = set
                    .bindings
                    .iter()
                    .map(|binding| {
                        let desc_type = convert_descriptor_type(binding.descriptor_type);
                        log::debug!(
                            "  - binding {}: {:?} x{} ({})",
                            binding.binding,
                            desc_type,
                            binding.count,
                            binding.name
                        );
                        vk::DescriptorSetLayoutBinding {
                            binding: binding.binding,
                            descriptor_type: desc_type,
                            descriptor_count: binding.count,
                            stage_flags: stage,
                            ..Default::default()
                        }
                    })
                    .collect();

                reflection.descriptor_sets.insert(set.set, bindings);
            }
        }

        // Extract vertex inputs
        if stage == vk::ShaderStageFlags::VERTEX {
            if let Ok(inputs) = module.enumerate_input_variables(None) {
                for var in inputs {
                    let format = convert_format(var.format);
                    log::debug!(
                        "[Shader Reflection] VERTEX input: location={}, format={:?} ({})",
                        var.location,
                        format,
                        var.name
                    );
                    reflection
                        .input_attributes
                        .push(vk::VertexInputAttributeDescription {
                            location: var.location,
                            binding: var.location,
                            format,
                            offset: 0,
                        });
                }
            }
        }

        // Extract fragment outputs
        if stage == vk::ShaderStageFlags::FRAGMENT {
            if let Ok(outputs) = module.enumerate_output_variables(None) {
                for var in outputs {
                    let format = convert_format(var.format);
                    log::debug!(
                        "[Shader Reflection] FRAGMENT output: location={}, format={:?}",
                        var.location,
                        format
                    );
                    reflection.output_attachment_formats.push(format);
                }
            }
        }

        // Log summary
        log::info!(
            "[Shader Reflection] {} shader: {} push constants, {} descriptor sets, {} inputs",
            stage_name,
            reflection.push_constants.len(),
            reflection.descriptor_sets.len(),
            reflection.input_attributes.len()
        );

        Ok(reflection)
    }

    /// Stub implementation when shader_reflection feature is disabled.
    /// Returns default empty reflection.
    #[cfg(not(feature = "shader_reflection"))]
    pub fn reflect(_code: &[u8], stage: vk::ShaderStageFlags) -> Result<Self> {
        log::warn!("ShaderReflection::reflect called without shader_reflection feature enabled - returning empty reflection");
        Ok(Self {
            stage,
            ..Default::default()
        })
    }

    /// Format a human-readable summary of shader resources
    pub fn format_summary(&self) -> String {
        let stage_name = match self.stage {
            vk::ShaderStageFlags::VERTEX => "VERTEX",
            vk::ShaderStageFlags::FRAGMENT => "FRAGMENT",
            _ => "OTHER",
        };

        let mut lines = vec![format!("{} shader resources:", stage_name)];

        for pc in &self.push_constants {
            lines.push(format!(
                "  Push constant: offset={}, size={}",
                pc.offset, pc.size
            ));
        }

        for set_idx in self.descriptor_sets.keys() {
            lines.push(format!("  Descriptor set {set_idx}"));
        }

        for attr in &self.input_attributes {
            lines.push(format!(
                "  Input: location={}, format={:?}",
                attr.location, attr.format
            ));
        }

        lines.join("\n")
    }
}

#[cfg(feature = "shader_reflection")]
fn convert_descriptor_type(
    ty: spirv_reflect::types::descriptor::ReflectDescriptorType,
) -> vk::DescriptorType {
    use spirv_reflect::types::descriptor::ReflectDescriptorType as Ty;
    match ty {
        Ty::Sampler => vk::DescriptorType::SAMPLER,
        Ty::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        Ty::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        Ty::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        Ty::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        Ty::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        Ty::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        Ty::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        Ty::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        Ty::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        Ty::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
        Ty::AccelerationStructureNV => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        _ => vk::DescriptorType::SAMPLER,
    }
}

#[cfg(feature = "shader_reflection")]
fn convert_format(format: spirv_reflect::types::ReflectFormat) -> vk::Format {
    use spirv_reflect::types::ReflectFormat as Fmt;
    match format {
        Fmt::R32G32B32A32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
        Fmt::R32G32B32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
        Fmt::R32G32_SFLOAT => vk::Format::R32G32_SFLOAT,
        Fmt::R32_SFLOAT => vk::Format::R32_SFLOAT,
        Fmt::R32G32B32A32_UINT => vk::Format::R32G32B32A32_UINT,
        Fmt::R32G32B32_UINT => vk::Format::R32G32B32_UINT,
        Fmt::R32G32_UINT => vk::Format::R32G32_UINT,
        Fmt::R32_UINT => vk::Format::R32_UINT,
        Fmt::R32G32B32A32_SINT => vk::Format::R32G32B32A32_SINT,
        Fmt::R32G32B32_SINT => vk::Format::R32G32B32_SINT,
        Fmt::R32G32_SINT => vk::Format::R32G32_SINT,
        Fmt::R32_SINT => vk::Format::R32_SINT,
        _ => vk::Format::R32G32B32A32_SFLOAT,
    }
}

/// Shader module wrapper holding reflection and entry-point metadata.
pub struct ShaderModule {
    pub module: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub entry_point: CString,
    pub reflection: ShaderReflection,
}

impl ShaderModule {
    pub fn load(
        device: &Arc<ash::Device>,
        path: impl AsRef<Path>,
        stage: vk::ShaderStageFlags,
    ) -> Result<Self> {
        let code = fs::read(path.as_ref()).map_err(|e| {
            AshError::VulkanError(format!("Failed to read shader {:?}: {e}", path.as_ref()))
        })?;

        Self::load_from_bytes(device, &code, stage)
    }

    pub fn load_from_bytes(
        device: &Arc<ash::Device>,
        code: &[u8],
        stage: vk::ShaderStageFlags,
    ) -> Result<Self> {
        if code.len() % 4 != 0 {
            return Err(AshError::VulkanError(
                "Shader size must be multiple of 4".to_string(),
            ));
        }

        let reflection = ShaderReflection::reflect(code, stage)?;

        let code_u32 = ash::util::read_spv(&mut Cursor::new(code))
            .map_err(|e| AshError::VulkanError(format!("Failed to parse SPIR-V: {e}")))?;

        let module = unsafe {
            let create_info = vk::ShaderModuleCreateInfo::default().code(&code_u32);
            device
                .create_shader_module(&create_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create shader module: {e}"))
                })?
        };

        Ok(Self {
            module,
            stage,
            entry_point: CString::new("main").unwrap(),
            reflection,
        })
    }

    pub fn stage_info(&self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage)
            .module(self.module)
            .name(&self.entry_point)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        // ShaderModule instances are usually cached and destroyed by the owner.
        // The actual Vulkan shader module destruction should be handled externally.
    }
}

/// Convenience loader returning just the shader module handle.
/// The caller should create the PipelineShaderStageCreateInfo themselves
/// since it requires references that must outlive the usage.
pub fn load_shader_module(device: &ash::Device, path: &str) -> Result<vk::ShaderModule> {
    let code = fs::read(path)
        .map_err(|e| AshError::VulkanError(format!("Failed to read shader {path}: {e}")))?;

    if code.len() % 4 != 0 {
        return Err(AshError::VulkanError(format!(
            "Shader {path} size must be multiple of 4"
        )));
    }

    let code_u32 =
        unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4) };

    let module = unsafe {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code_u32);
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| AshError::VulkanError(format!("Failed to create shader module: {e}")))?
    };

    Ok(module)
}
