#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "include/structures.glsl"

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

// Set 1: Bindless consolidated resources
layout(set = 1, binding = 0) uniform sampler2D textures[];

// Binding 1: Bindless Storage Buffers (for tints, per-material data, etc.)
// std430 for consistent layout between Rust and GLSL
layout(set = 1, binding = 1, std430) readonly buffer BindlessBuffer {
    vec4 data[];
} bindless_buffers[];


// Set 2: Environment (Skybox + ShadowMap + VSM)
layout(set = 2, binding = 3) uniform samplerCube skyboxMap;        // Optional: Skybox for reflections

layout(set = 2, binding = 5) uniform usampler2DArray vsmPageTable; // VSM Page Table Array (R32_UINT)
layout(set = 2, binding = 6) uniform sampler2D vsmPhysicalCache;   // VSM Physical Cache (R32_FLOAT)

// Set 3: No longer used (Forward+ migrated to BDA)
#define MAX_LIGHTS_PER_TILE 256

const float PI = 3.14159265359;

// Convert sRGB color to linear space for proper color handling
vec3 srgb_to_linear(vec3 color) {
    return mix(
        color / 12.92,
        pow((color + 0.055) / 1.055, vec3(2.4)),
        greaterThan(color, vec3(0.04045))
    );
}


// VSM Shadow Calculation - Virtual Shadow Maps with Clipmaps
// Uses page table lookup to find physical cache location
float VsmShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    FrameData frame = FrameData(push.frame_ptr);
    // For now, use a simple clipmap selection based on distance from camera
    // In a full implementation, this would use the ClipmapData uniform
    // to select the appropriate level based on world position
    
    // Calculate distance from camera for level selection
    vec3 worldPos = fragWorldPos;
    float distFromCamera = length(worldPos - frame.camera_pos.xyz);
    
    // Simple level selection (8 levels, exponentially spaced)
    // Level 0: 0-100m, Level 1: 100-200m, etc.
    int clipmapLayer = 0;
    float levelSize = 100.0;
    for (int i = 0; i < 8; i++) {
        if (distFromCamera < levelSize * float(i + 1)) {
            clipmapLayer = i;
            break;
        }
    }
    clipmapLayer = clamp(clipmapLayer, 0, 7);
    
    // 1. PROJECT & NORMALIZE TO [0,1]
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    
    // 2. EARLY REJECTION
    if (projCoords.z > 1.0) {
        return 0.0; // Outside shadow map = fully lit
    }
    
    // 3. CALCULATE VIRTUAL PAGE COORDINATES
    // Assume 16k virtual resolution with 128x128 pages = 128x128 page table
    const float virtualResolution = 16384.0;
    const float pageSize = 128.0;
    const float pageTableResolution = virtualResolution / pageSize; // 128
    
    vec2 virtualUV = clamp(projCoords.xy, 0.0, 1.0); // Simple clamp without custom bounds
    vec2 virtualPageFloat = virtualUV * pageTableResolution;
    ivec2 virtualPage = ivec2(virtualPageFloat);
    
    // 4. SAMPLE PAGE TABLE TO GET PHYSICAL PAGE (with layer)
    vec3 pageTableCoord = vec3((vec2(virtualPage) + 0.5) / pageTableResolution, float(clipmapLayer));
    uint packedPhysical = texture(vsmPageTable, pageTableCoord).r;
    
    // Check if page is allocated (0xFFFFFFFF = invalid)
    const uint INVALID_PAGE = 0xFFFFFFFFu;
    if (packedPhysical == INVALID_PAGE) {
        return 0.0; // Page not allocated = fully lit (no shadow data)
    }
    
    // 5. UNPACK PHYSICAL COORDINATES
    uint physical_x = packedPhysical & 0xFFFFu;
    uint physical_y = packedPhysical >> 16u;
    
    // 6. CALCULATE PHYSICAL UV
    // Physical cache is 4096x4096 with 128x128 pages = 32x32 pages
    const float physicalResolution = 4096.0;
    vec2 localUV = fract(virtualPageFloat); // UV within the page [0,1]
    vec2 physicalPageBase = vec2(float(physical_x), float(physical_y)) * pageSize;
    vec2 physicalPixel = physicalPageBase + localUV * pageSize;
    vec2 physicalUV = physicalPixel / physicalResolution;
    
    // 7. ADAPTIVE BIAS (Slope-Scaled)
    float cosAngle = clamp(dot(normal, lightDir), 0.0, 1.0);
    float NdotL = cosAngle;
    float slopeBias = clamp(0.005 * tan(acos(NdotL)), 0.0, 0.01);
    float bias = max(slopeBias, 0.0002);
    
    // 8. VARIANCE SHADOW MAPPING (Chebyshev's Inequality)
    // Sample variance moments (depth, depth^2) from physical cache
    vec2 moments = texture(vsmPhysicalCache, physicalUV).rg;
    
    float currentDepth = projCoords.z;
    
    // If current depth is closer than mean, fully lit
    if (currentDepth <= moments.x) {
        return 0.0; // No shadow
    }
    
    // Calculate variance and use Chebyshev's inequality
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, 0.00002); // Prevent division by zero
    
    float d = currentDepth - moments.x;
    float p_max = variance / (variance + d * d);
    
    // Apply light bleeding reduction (optional, helps with artifacts)
    float lightBleedingReduction = 0.2;
    p_max = clamp((p_max - lightBleedingReduction) / (1.0 - lightBleedingReduction), 0.0, 1.0);
    
    // Return shadow factor (1.0 = fully shadowed, 0.0 = fully lit)
    // Invert p_max because it represents visibility probability
    return 1.0 - p_max;
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // VSM by default
    return VsmShadowCalculation(fragPosLightSpace, normal, lightDir);
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

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================================================
// AMBIENT CALCULATION - RAGE HEMISPHERE
// ============================================================================

vec3 calculateHemisphereAmbient(vec3 normal, vec3 albedo) {
    FrameData frame = FrameData(push.frame_ptr);
    // Blend between sky and ground based on normal.y
    float skyFactor = normal.y * 0.5 + 0.5;
    
    vec3 skyContribution = frame.scene_lighting.ambient.sky_color.xyz * skyFactor;
    vec3 groundContribution = frame.scene_lighting.ambient.ground_color.xyz * (1.0 - skyFactor);
    
    vec3 ambientColor = (skyContribution + groundContribution) * frame.scene_lighting.ambient.sky_color.w;
    
    return ambientColor * albedo;
}

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
    
    // Shadows
    float shadow = 0.0;
    if (frame.scene_lighting.directional.direction.w > 0.5) {
        shadow = ShadowCalculation(fragPosLightSpace, N, L);
    }
    
    vec3 radiance = frame.scene_lighting.directional.color_intensity.rgb * frame.scene_lighting.directional.color_intensity.w;
    
    return (diffuse + specular) * radiance * NdotL * (1.0 - shadow);
}

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
        ? texture(textures[nonuniformEXT(base_color_idx)], fragUV)
        : vec4(1.0);
    // Apply sRGB-to-linear conversion only for texture samples (base_color_factor is already linear)
    vec3 baseSampleLinear = base_color_idx >= 0 ? srgb_to_linear(baseSample.rgb) : baseSample.rgb;
    vec3 baseColor = baseSampleLinear * base_color_factor.rgb * fragColor;

    // Apply tint from bindless storage buffer if available (Example 06 pattern)
    if (mat.tint_index >= 0) {
        vec4 tint = bindless_buffers[nonuniformEXT(mat.tint_index)].data[0];
        baseColor *= tint.rgb;
    }
    
    // Alpha discard
    if (baseSample.a * base_color_factor.a < mat.alpha_cutoff) {
        discard;
    }

    // Tangent-based Normal Mapping
    vec3 N = normalize(fragNormal);
    vec3 T_raw = fragTangent.xyz;
    vec3 T = length(T_raw) > 0.001 ? normalize(T_raw) : vec3(1.0, 0.0, 0.0);
    
    T = normalize(T - dot(T, N) * N);
    
    // Note: gl_FrontFacing check removed - redundant with backface culling enabled
    
    vec3 B = cross(N, T) * fragTangent.w;
    mat3 TBN = mat3(T, B, N);
    
    vec3 normal = N;
    int normal_idx = mat.texture_indices.y;
    float normal_scale = mat.parameters.w;

    // Normal Mapping
    if (normal_idx >= 0) {
        vec3 mapSample = texture(textures[nonuniformEXT(normal_idx)], fragUV).xyz;
        if (length(mapSample) > 0.001) {
            vec3 mapNormal = mapSample * 2.0 - 1.0;
            mapNormal.xy *= normal_scale;
            vec3 mapDir = TBN * mapNormal;
            if (length(mapDir) > 0.001) {
                normal = normalize(mapDir);
            }
        }
    }

    // ============================================================================
    // MATERIAL PARAMETERS
    // ============================================================================
    
    // Extract material parameters
    float metallic = mat.parameters.x;
    float roughness = mat.parameters.y;
    roughness = max(roughness, 0.04);
    
    // Metallic/Roughness Map
    int mr_idx = mat.texture_indices.z;
    if (mr_idx >= 0) {
        vec4 mrSample = texture(textures[nonuniformEXT(mr_idx)], fragUV);
        metallic = metallic * mrSample.b;
        roughness = max(roughness * mrSample.g, 0.04);
    }
    
    // Apply geometric AA to reduce specular aliasing from detailed normal maps
    roughness = adjust_roughness_geometric_aa(roughness, normal);

    // Ambient occlusion
    float occlusion = 1.0;
    int occ_idx = mat.texture_indices.w;
    float occ_strength = mat.parameters.z;

    // Occlusion Map
    if (occ_idx >= 0) {
        occlusion = mix(1.0, texture(textures[nonuniformEXT(occ_idx)], fragUV).r, occ_strength);
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
        
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * baseColor;
        
        // Accumulate this light's contribution
        vec3 radiance = light.color.rgb * light.color.a; // intensity in alpha
        Lo += (diffuse + specular) * radiance * NdotL * attenuation;
    }
    
    // Ambient Calculation - Hemisphere Fallback
    vec3 ambient = calculateHemisphereAmbient(normal, baseColor) * occlusion;
    
    // Layer 2: Global Directional Light
    vec3 directional = calculateDirectionalLight(
        normal, viewDir, baseColor, metallic, roughness, fragPosLightSpace
    );

    
    // Emissive
    int emissive_idx = mat.emissive_texture_index;
    vec3 emissive = mat.emissive_factor.rgb;
    if (emissive_idx >= 0) {
        emissive *= texture(textures[nonuniformEXT(emissive_idx)], fragUV).rgb;
    }

    // Combine: Ambient + Directional + Dynamic(Lo) + Emissive
    vec3 color = ambient + directional + Lo + emissive;

    // Final Output
    outColor = vec4(color, 1.0);
    outNormal = vec4(normal * 0.5 + 0.5, 1.0);
    outAlbedo = vec4(baseColor, 1.0);
    outMotion = motionVector;

    // Debug Path Visualization - Only compiled when debug_visualization feature is enabled
    #ifdef DEBUG_VISUALIZATION
    if (push.debug_visualization_enabled == 1) {
        // debug_path values: 1=GPU, 2=Legacy, 3=Albedo, 4=Normal, 5=Metallic, 6=Roughness, 7=Lighting
        if (push.debug_path == 1) { // GPU-Driven Path
            outColor = mix(outColor, vec4(0.0, 0.0, 1.0, 1.0), 0.3); // Blue tint
        } else if (push.debug_path == 2) { // Legacy Path
            outColor = mix(outColor, vec4(0.0, 1.0, 0.0, 1.0), 0.3); // Green tint
        } else if (push.debug_path == 3) { // Albedo
            outColor = vec4(baseColor, 1.0);
        } else if (push.debug_path == 4) { // Normal
            outColor = vec4(normal * 0.5 + 0.5, 1.0);
        } else if (push.debug_path == 5) { // Metallic
            outColor = vec4(vec3(metallic), 1.0);
        } else if (push.debug_path == 6) { // Roughness
            outColor = vec4(vec3(roughness), 1.0);
        } else if (push.debug_path == 7) { // Lighting Only
            outColor = vec4(ambient + directional + Lo, 1.0);
        }
    }
    #endif
}
