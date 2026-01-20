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

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 prev_view_proj;
    mat4 light_space_matrix;
    mat4 normal_matrix;
    vec4 camera_pos;
    SceneLighting scene_lighting;
} mvp;

struct MaterialUniform {
    vec4 base_color_factor;
    vec4 emissive_factor;
    vec4 parameters; // x: metallic, y: roughness, z: occlusion, w: normal_scale
    ivec4 texture_indices; // x: base, y: normal, z: mr, w: occlusion
    int emissive_texture_index;
    int tint_index;
    float alpha_cutoff;
    float _padding;
};



// Set 1: Bindless consolidated resources
layout(set = 1, binding = 0) uniform sampler2D textures[];
layout(std430, set = 1, binding = 1) readonly buffer MaterialBuffer {
    MaterialUniform materials[];
} material_buffers[];

// Tint buffer (still used by some parts, but consolidated to Set 1 if needed - 
// however Renderer doesn't seem to bind separate tint buffers in BindlessManager yet)
// Let's keep it in Set 2 for now IF Renderer still binds it there, 
// but wait, DescriptorManager Set 2 is Environment.
// Tints SHOULD be in Bindless (Set 1) if they are storage buffers.
// For now, I'll rely on MaterialUniform's fields.

// Set 2: Environment (IBL + Skybox + ShadowMap + VSM)
layout(set = 2, binding = 0) uniform samplerCube irradianceMap;   // Diffuse IBL
layout(set = 2, binding = 1) uniform samplerCube prefilterMap;     // Specular IBL  
layout(set = 2, binding = 2) uniform sampler2D brdfLUT;            // BRDF LUT texture
layout(set = 2, binding = 3) uniform samplerCube skyboxMap;        // Optional: Skybox for reflections

layout(set = 2, binding = 5) uniform usampler2DArray vsmPageTable; // VSM Page Table Array (R32_UINT)
layout(set = 2, binding = 6) uniform sampler2D vsmPhysicalCache;   // VSM Physical Cache (R32_FLOAT)

// Set 3: Forward+ Lighting (Modern tile-based deferred lighting)
#define MAX_LIGHTS_PER_TILE 256

struct Light {
    vec4 position;   // xyz = position, w = radius
    vec4 color;      // rgb = color, a = intensity
    vec4 direction;  // xyz = direction (for spot), w = type (0=point, 1=spot, 2=directional)
    vec4 params;     // x = innerConeAngle, y = outerConeAngle, z = falloff, w = enabled
};

layout(set = 3, binding = 0, std430) readonly buffer LightBuffer {
    Light lights[];
};

layout(set = 3, binding = 1, std430) readonly buffer TileLightIndices {
    uint tileData[];
};

layout(set = 3, binding = 2) uniform ForwardPlusInfo {
    uvec2 num_tiles;
    uint tile_size;
    uint _padding;
} fpInfo;

const float PI = 3.14159265359;

// Convert sRGB color to linear space for proper color handling
vec3 srgb_to_linear(vec3 color) {
    return mix(
        color / 12.92,
        pow((color + 0.055) / 1.055, vec3(2.4)),
        greaterThan(color, vec3(0.04045))
    );\
}

// VSM Shadow Calculation - Virtual Shadow Maps with Clipmaps
// Uses page table lookup to find physical cache location
float VsmShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // For now, use a simple clipmap selection based on distance from camera
    // In a full implementation, this would use the ClipmapData uniform
    // to select the appropriate level based on world position
    
    // Calculate distance from camera for level selection
    vec3 worldPos = fragWorldPos;
    float distFromCamera = length(worldPos - mvp.camera_pos.xyz);
    
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
    if (projCoords.x < push.uv_min || projCoords.x > push.uv_max ||
        projCoords.y < push.uv_min || projCoords.y > push.uv_max ||
        projCoords.z > 1.0) {
        return 0.0; // Outside shadow map = fully lit
    }
    
    // 3. CALCULATE VIRTUAL PAGE COORDINATES
    // Assume 16k virtual resolution with 128x128 pages = 128x128 page table
    const float virtualResolution = 16384.0;
    const float pageSize = 128.0;
    const float pageTableResolution = virtualResolution / pageSize; // 128
    
    vec2 virtualUV = clamp(projCoords.xy, vec2(push.uv_min), vec2(push.uv_max));
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
    // Use slope-scaled bias to prevent acne on steep surfaces
    // standard bias: max(0.005 * (1.0 - dot), 0.0005)
    // improved: clamp(factor * tan(acos(ndotl)), min, max)
    float NdotL = cosAngle;
    float slopeBias = clamp(0.005 * tan(acos(NdotL)), 0.0, 0.01);
    float bias = max(slopeBias, 0.0002);
    
    // 8. SAMPLE PHYSICAL CACHE WITH VOGEL DISK (SMRT-like Filter)
    float shadow = 0.0;
    float texelSize = 1.0 / physicalResolution;
    float currentDepth = projCoords.z;
    
    const int SAMPLE_COUNT = 16;
    const float GOLDEN_ANGLE = 2.4; // Radians
    
    // Interleaved Gradient Noise for rotation
    float noise = fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
    float rotation = noise * 6.283185;
    float sinRot = sin(rotation);
    float cosRot = cos(rotation);
    
    // Filter radius (tunable)
    float filterRadius = 2.0;

    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        // Generate Vogel Sample Offset
        float r = sqrt(float(i) + 0.5) / sqrt(float(SAMPLE_COUNT));
        float theta = i * GOLDEN_ANGLE + rotation;
        
        // Rotate the offset
        vec2 offset = vec2(cos(theta), sin(theta)) * r * filterRadius * texelSize;
        
        float vsmDepth = texture(vsmPhysicalCache, physicalUV + offset).r;
        
        if (vsmDepth == 1.0) {
             // 1.0 usually means "clear value" / far plane in shadow map
             // If shadow map is cleared to 1.0, and our currentDepth < 1.0, we are lit?
             // Need to check clear color. VSM clears to 1.0.
             shadow += 1.0; 
        } else {
             shadow += (currentDepth - bias) > vsmDepth ? 1.0 : 0.0;
        }
    }
    
    // Invert shadow sum (logic above was: if depth > map ? 1.0 (SHADOW))
    // Wait.
    // Standard: if (currentDepth > closestDepth + bias) Shadow = 1.0;
    // My logic: shadow += (currentDepth - bias) > pcfDepth ? 1.0 : 0.0;
    // So "shadow" accumulates Occlusion.
    // 0 = Lit, 16 = Fully Occluded.
    
    // We want to return visibility (1.0 = Lit, 0.0 = Shadow).
    // So if sum is 16, Visibility is 0.
    
    return 1.0 - (shadow / float(SAMPLE_COUNT));
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // Use VSM by default
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
    // Blend between sky and ground based on normal.y
    float skyFactor = normal.y * 0.5 + 0.5;
    
    vec3 skyContribution = mvp.scene_lighting.ambient.sky_color.xyz * skyFactor;
    vec3 groundContribution = mvp.scene_lighting.ambient.ground_color.xyz * (1.0 - skyFactor);
    
    vec3 ambientColor = (skyContribution + groundContribution) * mvp.scene_lighting.ambient.sky_color.w;
    
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
    vec3 L = -normalize(mvp.scene_lighting.directional.direction.xyz);
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
    if (mvp.scene_lighting.directional.direction.w > 0.5) {
        shadow = ShadowCalculation(fragPosLightSpace, N, L);
    }
    
    vec3 radiance = mvp.scene_lighting.directional.color_intensity.rgb * mvp.scene_lighting.directional.color_intensity.w;
    
    return (diffuse + specular) * radiance * NdotL * (1.0 - shadow);
}

void main() {
    // Extract actual index from handle (lower 16 bits)
    uint actual_material_index = push.material_index & 0xFFFFu;
    MaterialUniform mat = material_buffers[nonuniformEXT(push.material_buffer_index)].materials[actual_material_index];

    vec3 viewDir = normalize(mvp.camera_pos.xyz - fragWorldPos);

    // Sample base color
    int base_color_idx = mat.texture_indices.x;
    vec4 base_color_factor = mat.base_color_factor;

    vec4 baseSample = base_color_idx >= 0
        ? texture(textures[nonuniformEXT(base_color_idx)], fragUV)
        : vec4(1.0);
    // Apply sRGB-to-linear conversion only for texture samples (base_color_factor is already linear)
    vec3 baseSampleLinear = base_color_idx >= 0 ? srgb_to_linear(baseSample.rgb) : baseSample.rgb;
    vec3 baseColor = baseSampleLinear * base_color_factor.rgb * fragColor;
    
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

    if (occ_idx >= 0) {
        occlusion = mix(1.0, texture(textures[nonuniformEXT(occ_idx)], fragUV).r, occ_strength);
    }

    // PBR base reflectance
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    float NdotV = max(dot(normal, viewDir), 0.001);

    // ============================================================================
    // MODERN FORWARD+ LIGHTING (Tile-Based Deferred)
    // ============================================================================
    
    // Calculate tile index for this fragment
    uvec2 tileID = uvec2(gl_FragCoord.xy) / fpInfo.tile_size;
    uint tileIndex = tileID.y * fpInfo.num_tiles.x + tileID.x;
    uint tileOffset = tileIndex * (MAX_LIGHTS_PER_TILE + 1);
    
    // Get light count for this tile (first element)
    uint lightCount = min(tileData[tileOffset], MAX_LIGHTS_PER_TILE);
    
    // Accumulated lighting
    vec3 Lo = vec3(0.0);
    
    // Iterate over all lights affecting this tile
    for (uint i = 0; i < lightCount; i++) {
        uint lightIdx = tileData[tileOffset + 1 + i];
        Light light = lights[lightIdx];
        
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
    
    // ============================================================================
    // ============================================================================
    // IMAGE-BASED LIGHTING (IBL) + GLOBAL DIRECTIONAL
    // ============================================================================

    // Layer 1: IBL Ambient (Diffuse + Specular)
    // Diffuse IBL: Irradiance map provides diffuse ambient
    // Reuse F0 already calculated at line 324
    vec3 F = fresnel_schlick_roughness(max(dot(normal, viewDir), 0.0), F0, roughness);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    
    vec3 irradiance = texture(irradianceMap, normal).rgb;
    vec3 diffuseIBL = kD * irradiance * baseColor;
    
    // Specular IBL: Prefiltered environment map + BRDF LUT
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-viewDir, normal);
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF = texture(brdfLUT, vec2(max(dot(normal, viewDir), 0.0), roughness)).rg;
    vec3 specularIBL = prefilteredColor * (F * envBRDF.x + envBRDF.y);
    
    vec3 ambient = (diffuseIBL + specularIBL) * occlusion;
    
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

    // Debug Path Visualization - Only compiled when debug_visualization feature is enabled
    // Debug Path Visualization - Only compiled when debug_visualization feature is enabled
    #ifdef DEBUG_VISUALIZATION
    if (push.debug_visualization_enabled == 1) {
        // debug_path values: 1=GPU, 2=Legacy, 3=Albedo, 4=Normal, 5=Metallic, 6=Roughness, 7=Lighting
        if (push.debug_path == 1) { // GPU-Driven Path
            color = mix(color, vec3(0.0, 0.0, 1.0), 0.3); // Blue tint
        } else if (push.debug_path == 2) { // Legacy Path
            color = mix(color, vec3(0.0, 1.0, 0.0), 0.3); // Green tint
        } else if (push.debug_path == 3) { // Albedo
            color = baseColor;
        } else if (push.debug_path == 4) { // Normal
            color = normal * 0.5 + 0.5;
        } else if (push.debug_path == 5) { // Metallic
            color = vec3(metallic);
        } else if (push.debug_path == 6) { // Roughness
            color = vec3(roughness);
        } else if (push.debug_path == 7) { // Lighting Only
            // Show accumulated light without albedo modulation
            // Recalculate basic lighting sum for visualization
            color = ambient + directional + Lo;
        }
    }
    #endif
    
    outColor = vec4(color, 1.0);
    outNormal = vec4(normal, 1.0);
    outAlbedo = vec4(baseColor, 1.0);
    outMotion = motionVector;
}
