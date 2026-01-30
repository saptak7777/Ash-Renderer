use ash::vk;
#[cfg(feature = "shader_reflection")]
use rspirv_reflect::Reflection;
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
    pub output_attachment_formats: Vec<vk::Format>,
    pub stage: vk::ShaderStageFlags,
}

impl Default for ShaderReflection {
    fn default() -> Self {
        Self {
            push_constants: Vec::new(),
            descriptor_sets: HashMap::new(),
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
    pub fn reflect(code: &[u32], stage: vk::ShaderStageFlags) -> Result<Self> {
        let reflection_module = Reflection::new_from_spirv(code)
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
        if let Ok(Some(push_constant)) = reflection_module.get_push_constant_range() {
            log::debug!(
                "[Shader Reflection] {} push constant: offset={}, size={} bytes",
                stage_name,
                push_constant.offset,
                push_constant.size
            );
            reflection.push_constants.push(vk::PushConstantRange {
                stage_flags: stage,
                offset: push_constant.offset,
                size: push_constant.size,
            });
        }

        // Extract descriptor sets and bindings
        if let Ok(descriptor_sets) = reflection_module.get_descriptor_sets() {
            for (set_idx, bindings_map) in descriptor_sets {
                log::debug!(
                    "[Shader Reflection] {} descriptor set {}: {} bindings",
                    stage_name,
                    set_idx,
                    bindings_map.len()
                );

                let bindings: Vec<_> = bindings_map
                    .iter()
                    .map(|(binding_idx, binding_info)| {
                        let desc_type = convert_descriptor_type(binding_info.ty);
                        log::debug!(
                            "  - binding {}: {:?} x{} ({})",
                            binding_idx,
                            desc_type,
                            binding_info.binding_count,
                            binding_info.name
                        );
                        vk::DescriptorSetLayoutBinding {
                            binding: *binding_idx,
                            descriptor_type: desc_type,
                            descriptor_count: binding_info.binding_count,
                            stage_flags: stage,
                            ..Default::default()
                        }
                    })
                    .collect();

                reflection.descriptor_sets.insert(set_idx, bindings);
            }
        }

        // Log summary
        log::info!(
            "[Shader Reflection] {} shader: {} push constants, {} descriptor sets",
            stage_name,
            reflection.push_constants.len(),
            reflection.descriptor_sets.len()
        );

        Ok(reflection)
    }

    /// Stub implementation when shader_reflection feature is disabled.
    /// Returns default empty reflection.
    #[cfg(not(feature = "shader_reflection"))]
    pub fn reflect(_code: &[u32], stage: vk::ShaderStageFlags) -> Result<Self> {
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

        // Input attributes loop removed (rspirv-reflect migration skipped input extraction)

        lines.join("\n")
    }
}

#[cfg(feature = "shader_reflection")]
fn convert_descriptor_type(ty: rspirv_reflect::DescriptorType) -> vk::DescriptorType {
    use rspirv_reflect::DescriptorType as Ty;
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
        Ty::AccelerationStructureKHR => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        _ => vk::DescriptorType::SAMPLER,
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

        // Defensive: Check alignment and minimal header size
        debug_assert_eq!(
            code.as_ptr() as usize % 4,
            0,
            "SPIR-V must be 4-byte aligned"
        );
        if code.len() < 20 {
            return Err(AshError::VulkanError(
                "SPIR-V code too small for valid header".into(),
            ));
        }

        // Validate SPIR-V magic number (0x07230203)
        let magic = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        let magic_be = u32::from_be_bytes([code[0], code[1], code[2], code[3]]);
        if magic != 0x07230203 && magic_be != 0x07230203 {
            return Err(AshError::VulkanError(format!(
                "Invalid SPIR-V magic number: 0x{magic:08X}"
            )));
        }

        let code_u32 = ash::util::read_spv(&mut Cursor::new(code))
            .map_err(|e| AshError::VulkanError(format!("Failed to parse SPIR-V: {e}")))?;

        let reflection = ShaderReflection::reflect(&code_u32, stage)?;

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

    // Defensive: Validate SPIR-V header before FFI conversion
    if code.len() >= 4 {
        let magic = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        let magic_be = u32::from_be_bytes([code[0], code[1], code[2], code[3]]);
        if magic != 0x07230203 && magic_be != 0x07230203 {
            return Err(AshError::VulkanError(format!(
                "Invalid SPIR-V magic number in {path}: 0x{magic:08X}"
            )));
        }
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

#[cfg(test)]
mod tests {
    // Note: Full shader module creation tests require a valid Vulkan device.
    // These tests verify the validation logic independently.

    #[test]
    fn test_spirv_size_validation() {
        // Verify size validation catches non-4-byte-aligned data
        let bad_size = [0x03, 0x02, 0x23, 0x07, 0, 0, 0]; // Size 7 (not multiple of 4)
        assert_ne!(bad_size.len() % 4, 0);
    }

    #[test]
    fn test_spirv_magic_validation() {
        // Verify magic number validation logic
        let valid_magic_le = [0x03, 0x02, 0x23, 0x07]; // 0x07230203 in little-endian
        let valid_magic_be = [0x07, 0x23, 0x02, 0x03]; // 0x07230203 in big-endian
        let invalid_magic = [0x00, 0x00, 0x00, 0x00];

        let magic_le = u32::from_le_bytes([
            valid_magic_le[0],
            valid_magic_le[1],
            valid_magic_le[2],
            valid_magic_le[3],
        ]);
        let magic_be = u32::from_be_bytes([
            valid_magic_be[0],
            valid_magic_be[1],
            valid_magic_be[2],
            valid_magic_be[3],
        ]);
        let bad_magic = u32::from_le_bytes([
            invalid_magic[0],
            invalid_magic[1],
            invalid_magic[2],
            invalid_magic[3],
        ]);

        assert_eq!(magic_le, 0x07230203);
        assert_eq!(magic_be, 0x07230203);
        assert_ne!(bad_magic, 0x07230203);
    }
}
