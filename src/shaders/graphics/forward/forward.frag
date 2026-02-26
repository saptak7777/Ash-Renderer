#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "interop/structures.glsl"
#include "interop/bindings.glsl"
#include "common/pbr_utils.glsl"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) centroid in vec3 fragNormal;
layout(location = 3) sample in vec3 fragWorldPos;
layout(location = 4) in vec4 fragPosLightSpace;
layout(location = 5) in vec4 fragTangent;
layout(location = 6) in vec2 motionVector;
layout(location = 7) flat in uint fragInstanceIndex;

// Geometric Specular AA (Toksvig/Kaplanyan method)
// Adjusts roughness based on normal map variance to prevent specular aliasing
float adjust_roughness_geometric_aa(float roughness, vec3 normal) {
    // Measure normal variation using screen-space derivatives
    vec3 dndu = dFdx(normal);
    vec3 dndv = dFdy(normal);
    float variance = dot(dndu, dndu) + dot(dndv, dndv);
    
    // Toksvig AA formula: increases roughness on high-frequency normals
    float kernelRoughness = min(2.0 * variance, 1.0);
    return sqrt(roughness * roughness + kernelRoughness);
}

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec2 outMotion;

// Set 1: Reserved for future use (Materials or other)
// layout(set = 1, binding = 0) ...
const uint MAX_LIGHTS_PER_TILE = 256;

// NOTE: No srgb_to_linear function here.
// Albedo textures are uploaded as vk::Format::R8G8B8A8_SRGB.
// The Vulkan sampler automatically linearizes them on read.
// Any manual conversion here would apply the curve twice, darkening albedo.


float SampleVSM(vec3 worldPos) {
    VsmGlobal u_Global = VsmGlobal(push.vsm_ptr);
    // 1. Project World -> Light Clip Space (Cascade 0 for now)
    vec4 shadowClip = u_Global.light_view_projections[0] * vec4(worldPos, 1.0);
    vec3 shadowNDC = shadowClip.xyz / shadowClip.w;
    vec2 shadowUV = shadowNDC.xy * 0.5 + 0.5;
    shadowUV.y = 1.0 - shadowUV.y; // Flip Y for Vulkan convention

    // 2. Check bounds
    if (any(lessThan(shadowUV, vec2(0.0))) || any(greaterThan(shadowUV, vec2(1.0)))) {
        return 1.0; // Outside shadow map -> Unshadowed
    }

    // Use bindless page table access (Sampled Image)
    ivec2 pageTableSize = textureSize(global_page_tables[nonuniformEXT(u_Global.page_table_index)], 0).xy;
    ivec2 pageCoord = ivec2(shadowUV * vec2(pageTableSize));
    
    // Read Page Entry (R32UI) from Layer 0 (Directional Light)
    uint pageEntry = texelFetch(global_page_tables[nonuniformEXT(u_Global.page_table_index)], ivec3(pageCoord, 0), 0).r;
    
    // 4. Check Residency
    if (pageEntry == 0xFFFFFFFFu) return 1.0; 

    // 5. Physical Address Translation
    // Unpack Physical Coord (16-bit X | 16-bit Y)
    uint pX = pageEntry & 0xFFFFu;
    uint pY = (pageEntry >> 16) & 0xFFFFu;
    
    // Fraction inside the page
    vec2 pageFract = fract(shadowUV * vec2(pageTableSize));
    ivec2 physicalTexel = ivec2(pX, pY) * 128 + ivec2(pageFract * 128.0);
    
    // 6. Sample VSM Moments (R32G32F: Depth, Depth^2)
    // Use bindless physical cache access (Sampled Image)
    vec2 moments = texelFetch(global_textures[nonuniformEXT(u_Global.physical_cache_index)], physicalTexel, 0).rg;
    
    // 7. Chebyshev's Inequality
    float currentDepth = shadowNDC.z;
    
    // Fully lit if closer than the average depth
    if (currentDepth <= moments.x) {
        return 1.0;
    }
    
    // Variance calculation with epsilon to prevent NaNs and light bleeding
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, 0.00001);
    
    float d = currentDepth - moments.x;
    float p_max = variance / (variance + d * d);
    
    // REVERSE-Z: Optional: Clamp p_max to prevent excessive light bleeding in high contrast areas
    return p_max;
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    return SampleVSM(fragWorldPos);
}

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometry_schlick_ggx_fast(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return geometry_schlick_ggx_fast(NdotV, k) * geometry_schlick_ggx_fast(NdotL, k);
}

vec3 fresnel_schlick_fast(float cosTheta, vec3 F0) {
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (1.0 - F0) * t5;
}

// Hemisphere ambient logic transitioned to IBL in pbr_utils.glsl

// ============================================================================
// DIRECTIONAL LIGHT CALCULATION
// ============================================================================

vec3 calculateDirectionalLight(
    vec3 N,
    vec3 V,
    vec3 albedo,
    float metallic,
    float roughness,
    vec4 fragPosLightSpace
) {
    FrameData frame = FrameData(push.frame_ptr);

    vec3 L = -normalize(frame.scene_lighting.directional.direction.xyz);
    vec3 H = normalize(V + L);
    
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnel_schlick_fast(max(dot(H, V), 0.0), F0);
    float D = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);
    
    vec3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 0.001);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;
    
    // Visibility (Shadows)
    float visibility = 1.0;
    if (frame.scene_lighting.directional.direction.w > 0.5) {
        visibility = ShadowCalculation(fragPosLightSpace, N, L);
    }
    
    vec3 radiance = frame.scene_lighting.directional.color_intensity.rgb * frame.scene_lighting.directional.color_intensity.w;
    
    return (diffuse + specular) * radiance * NdotL * visibility;
}

// IBL logic shifted to pbr_utils.glsl for cross-pass reuse

void main() {
    FrameData frame = FrameData(push.frame_ptr);
    // Extract actual index from handle (lower 16 bits)
    uint actual_material_index = push.material_index;
    
    // Modern BDA Material Pulling (Per-Instance)
    if (push.use_instancing == 1 && push.instance_ptr != 0) {
        InstanceBuffer instance_ctx = InstanceBuffer(push.instance_ptr);
        // gl_InstanceIndex is not available in fragment shader, use passed in index
        InstanceData instance = instance_ctx.instances[fragInstanceIndex];
        actual_material_index = instance.material_index;
    }
    
    actual_material_index &= 0xFFFFu;
    
    MaterialBuffer material_ctx = MaterialBuffer(push.material_ptr);
    MaterialData mat = material_ctx.materials[actual_material_index];

    vec3 viewDir = normalize(frame.camera_pos.xyz - fragWorldPos);

    // Sample base color
    int base_color_idx = mat.texture_indices.x;
    vec4 base_color_factor = mat.base_color_factor;

    // Sample base color texture
    vec4 baseSample = base_color_idx >= 0
        ? texture(global_textures[nonuniformEXT(base_color_idx)], fragUV)
        : vec4(1.0);
    // Hardware linearization: R8G8B8A8_SRGB format decodes sRGB→linear on the sampler.
    // Simply read the texel — no manual conversion needed or permitted.
    vec3 baseSampleLinear = baseSample.rgb;
    vec3 baseColor = baseSampleLinear * base_color_factor.rgb * fragColor;

    // Apply tint from bindless storage buffer if available (Example 06 pattern)
    if (mat.tint_index >= 0) {
        vec4 tint = bindless_buffers[nonuniformEXT(mat.tint_index)].data[0];
        baseColor *= tint.rgb;
    }
    
    // Alpha discard
    if ((mat.flags & MATERIAL_FLAG_ALPHA_TESTED) != 0u) {
        if (baseSample.a * base_color_factor.a < mat.alpha_cutoff) {
            discard;
        }
    }

    // Tangent-based Normal Mapping
    vec3 N = normalize(fragNormal);
    vec3 T_raw = fragTangent.xyz;
    vec3 T = length(T_raw) > 0.001 ? normalize(T_raw) : vec3(1.0, 0.0, 0.0);
    
    // Gram-Schmidt orthogonalization - Safety check for degenerate T
    if (dot(T, T) > 0.001) {
        T = normalize(T - dot(T, N) * N);
    }
    
    // Note: gl_FrontFacing check removed - redundant with backface culling enabled
    
    vec3 B = cross(N, T) * fragTangent.w;
    mat3 TBN = mat3(T, B, N);
    
    vec3 normal = N;
    int normal_idx = mat.texture_indices.y;
    float normal_scale = mat.parameters.w;

    // Normal Mapping
    // Normal Mapping
    if (normal_idx >= 0) {
        // BC5 Compression Store RG only. B is 0.0 or 1.0 depending on decoder, but we must reconstruct Z.
        // We assume 2-channel normal maps if Z is consistently 0 (or simply enforce reconstruction).
        // Since we aggressively optimize, we use Z-reconstruction for ALL normal maps to be safe.
        vec2 mapSampleXY = texture(global_textures[nonuniformEXT(normal_idx)], fragUV).xy;
        
        // Unpack from [0,1] to [-1,1]
        vec2 mapNormalXY = mapSampleXY * 2.0 - 1.0;
        mapNormalXY *= normal_scale;

        // Reconstruct Z: z = sqrt(1 - x^2 - y^2)
        // Clamp to prevent NaN if normal map is invalid/unnormalized
        float z2 = 1.0 - dot(mapNormalXY, mapNormalXY);
        float mapNormalZ = sqrt(max(z2, 0.0));

        vec3 mapNormal = vec3(mapNormalXY, mapNormalZ);

        vec3 mapDir = TBN * mapNormal;
        if (length(mapDir) > 0.001) {
            normal = normalize(mapDir);
        }
    }

    // ============================================================================
    // MATERIAL PARAMETERS
    // ============================================================================
    
    // Extract material parameters (Factors)
    float metallic = mat.parameters.x;
    float roughness = mat.parameters.y;
    
    // Metallic/Roughness Map (Texture Accumulation)
    int mr_idx = mat.texture_indices.z;
    if (mr_idx >= 0) {
        vec4 mrSample = texture(global_textures[nonuniformEXT(mr_idx)], fragUV);
        metallic *= mrSample.b;
        roughness *= mrSample.g;
    }

    // Safety clamps and geometric AA
    roughness = max(roughness, 0.04);
    roughness = adjust_roughness_geometric_aa(roughness, normal);

    // Ambient occlusion
    float occlusion = 1.0;
    int occ_idx = mat.texture_indices.w;
    float occ_strength = mat.parameters.z;

    // Occlusion Map
    if (occ_idx >= 0) {
        occlusion = mix(1.0, texture(global_textures[nonuniformEXT(occ_idx)], fragUV).r, occ_strength);
    }

    // PBR base reflectance
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    float NdotV = max(dot(normal, viewDir), 0.001);

    // ============================================================================
    // MODERN FORWARD+ LIGHTING (Tile-Based Deferred BDA)
    // ============================================================================
    LightBuffer lb = LightBuffer(push.light_ptr);
    TileIndexBuffer tib = TileIndexBuffer(push.tile_ptr);

    // Calculate tile index for this fragment
    uvec2 tileID = uvec2(gl_FragCoord.xy) / frame.scene_lighting.tile_size;
    uint tileIndex = tileID.y * frame.scene_lighting.num_tiles_x + tileID.x;
    uint tileOffset = tileIndex * (MAX_LIGHTS_PER_TILE + 1);
    
    // Get light count for this tile (first element)
    uint lightCount = min(tib.tileData[tileOffset], MAX_LIGHTS_PER_TILE);
    
    // Accumulated lighting
    vec3 Lo = vec3(0.0);
    
    // Iterate over all lights affecting this tile
    for (uint i = 0; i < lightCount; i++) {
        uint lightIdx = tib.tileData[tileOffset + 1 + i];
        Light light = lb.lights[lightIdx];
        
        // Skip disabled lights
        if (light.params.w < 0.5) continue;
        
        // Light type constants (aligned with Rust and compute shader)
        const uint LIGHT_TYPE_POINT = 0u;
        const uint LIGHT_TYPE_SPOT = 2u;
        
        uint lightType = uint(light.direction.w);
        vec3 L; // Light direction
        float attenuation = 1.0;
        
        // Point Light
        if (lightType == LIGHT_TYPE_POINT) {
            vec3 lightVec = light.position.xyz - fragWorldPos;
            float dist = length(lightVec);
            float radius = light.position.w;
            
            // Skip if outside radius
            if (dist > radius) continue;
            
            L = lightVec / dist;
            
            // Inverse square falloff with smooth cutoff
            float distRatio = dist / radius;
            attenuation = 1.0 / (dist * dist + 1.0);
            attenuation *= max(0.0, 1.0 - distRatio * distRatio);
        }
        // Spot Light
        else if (lightType == LIGHT_TYPE_SPOT) {
            vec3 lightVec = light.position.xyz - fragWorldPos;
            float dist = length(lightVec);
            float range = light.position.w;
            
            // Skip if outside range
            if (dist > range) continue;
            
            L = lightVec / dist;
            
            // Spot attenuation (cone falloff)
            vec3 lightDir = normalize(light.direction.xyz);
            float cosAngle = dot(-L, lightDir);
            float cosInner = light.params.x;
            float cosOuter = light.params.y;
            
            if (cosAngle < cosOuter) continue;
            
            float spotAttenuation = smoothstep(cosOuter, cosInner, cosAngle);
            
            // Distance attenuation
            float distRatio = dist / range;
            attenuation = 1.0 / (dist * dist + 1.0);
            attenuation *= max(0.0, 1.0 - distRatio * distRatio);
            attenuation *= spotAttenuation;
        }
        else {
            continue;
        }
        
        // PBR calculation for this light
        float NdotL = max(dot(normal, L), 0.0);
        if (NdotL <= 0.0) continue;
        
        vec3 H = normalize(viewDir + L);
        float NdotH = max(dot(normal, H), 0.0);
        float VdotH = max(dot(viewDir, H), 0.0);
        
        // Cook-Torrance BRDF
        float D = distribution_ggx(NdotH, roughness);
        float G = geometry_smith(NdotV, NdotL, roughness);
        vec3 F = fresnel_schlick_fast(VdotH, F0);
        
        vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
        
        // HDR-safe specular clamping (FP16 buffer limit)
        // Geometric AA already reduces fireflies, this prevents buffer overflow
        float specularMax = max(max(specular.r, specular.g), specular.b);
        if (specularMax > 65000.0) {
            specular *= 65000.0 / specularMax;
        }
        
        // Diffuse term must be normalized by PI (Energy Conservation)
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * baseColor / PI;
        
        // Accumulate this light's contribution
        vec3 radiance = light.color.rgb * light.color.a; // intensity in alpha
        Lo += (diffuse + specular) * radiance * NdotL * attenuation;
    }
    
    // Final Reflectance (F0) and Reflection Vector (R) for IBL
    vec3 R = reflect(-viewDir, normal);

    // Layer 1: Ambient (IBL)
    vec3 ambient = getIBLContribution(NdotV, normal, R, F0, roughness, baseColor, metallic, occlusion);
    ambient *= frame.scene_lighting.ibl_intensity; // Apply global intensity

    // Layer 2: Global Directional Light
    vec3 directional = calculateDirectionalLight(
        normal, viewDir, baseColor, metallic, roughness, fragPosLightSpace
    );
    // directional *= SampleVSM(fragWorldPos); // REMOVED: calculateDirectionalLight already calls ShadowCalculation

    // Emissive
    int emissive_idx = mat.emissive_texture_index;
    vec3 emissive = mat.emissive_factor.rgb;
    if (emissive_idx >= 0) {
        emissive *= texture(global_textures[nonuniformEXT(emissive_idx)], fragUV).rgb;
    }

    // Combine: IBL + Directional + Dynamic(Lo) + Emissive
    vec3 color = ambient + directional + Lo + emissive;

    // FP16 Safety Clamp: Max value is bounded to 65000.0 (near FP16 max of 65504.0)
    // This prevents NaN propagation and overflow in the bloom/resolve passes 
    // while preserving extreme high dynamic range for intense highlights.
    color = clamp(color, 0.0, 65000.0);

    // Final Output (Restored for SRGB standardized pipeline)
    outColor = vec4(color, 1.0);
    outNormal = vec4(normal * 0.5 + 0.5, 1.0);
    outAlbedo = vec4(baseColor, 1.0);
    outMotion = motionVector;

    // Debug Path Visualization (Neutralized for Tonemapper Trap)
    if (push.debug_mode > 0) {
        const vec3 gamma_fix = vec3(2.2);
        
        // Mode 1: Color by LOD (Error Metric)
        if (push.debug_mode == 1) {
             float error = 0.0;
             if (push.use_instancing == 1 && push.instance_ptr != 0) {
                 InstanceBuffer instance_ctx = InstanceBuffer(push.instance_ptr);
                 InstanceData instance = instance_ctx.instances[fragInstanceIndex];
                 error = instance.error_metric;
             }
             vec3 errorColor = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), error * 10.0);
             outColor = vec4(pow(errorColor, gamma_fix), 1.0);
        }
        // Mode 2: Color by Cluster/Instance ID
        else if (push.debug_mode == 2) {
             uint seed = fragInstanceIndex;
             seed = (seed ^ 61u) ^ (seed >> 16u);
             seed *= 9u;
             seed = seed ^ (seed >> 4u);
             seed *= 0x27d4eb2d;
             seed = seed ^ (seed >> 15u);
             float r = float(seed) * (1.0/4294967296.0);
             float g = float(seed * 16807u) * (1.0/4294967296.0);
             float b = float(seed * 48271u) * (1.0/4294967296.0);
             outColor = vec4(pow(vec3(r, g, b), gamma_fix), 1.0);
        }
    }
}
