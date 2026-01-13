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

// Edge detection helper for adaptive anti-aliasing
// DISABLED: Was causing darkening and cartoonish appearance
vec3 edge_denoise(vec3 color, vec3 worldPos) {
    return color;
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

// Set 2: Environment (IBL + Skybox + ShadowMap)
layout(set = 2, binding = 0) uniform samplerCube irradianceMap;   // Diffuse IBL
layout(set = 2, binding = 1) uniform samplerCube prefilterMap;     // Specular IBL  
layout(set = 2, binding = 2) uniform sampler2D brdfLUT;            // BRDF LUT texture
layout(set = 2, binding = 3) uniform samplerCube skyboxMap;        // Optional: Skybox for reflections
layout(set = 2, binding = 4) uniform sampler2D shadowMap;          // Moved from binding 0

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
    );
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    // CRITICAL FIX: Clamp projCoords to [0,1] range to prevent edge artifacts from PCF sampling
    // Without this, textureGather at boundaries reads outside the shadow map, causing glitchy overlaps
    projCoords = clamp(projCoords, vec3(0.0), vec3(1.0));
    
    float currentDepth = projCoords.z;
    
    // Adaptive bias based on surface slope relative to light direction
    float cosAngle = clamp(dot(normal, lightDir), 0.0, 1.0);
    float minBias = 0.0005;
    float maxBias = 0.005;
    float bias = max(maxBias * (1.0 - cosAngle), minBias);
    
    if(projCoords.z > 1.0)
        return 0.0;
    
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    vec2 uv = projCoords.xy;
    
    float shadow = 0.0;
    vec4 g0 = textureGather(shadowMap, uv + vec2(-1.0, -1.0) * texelSize);
    vec4 g1 = textureGather(shadowMap, uv + vec2( 1.0, -1.0) * texelSize);
    vec4 g2 = textureGather(shadowMap, uv + vec2(-1.0,  1.0) * texelSize);
    vec4 g3 = textureGather(shadowMap, uv + vec2( 1.0,  1.0) * texelSize);
    
    float compareDepth = currentDepth - bias;
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g0)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g1)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g2)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g3)), vec4(1.0));
    
    return shadow / 16.0;
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
        
        // Clamp extreme specular highlights
        float specularMax = max(max(specular.r, specular.g), specular.b);
        if (specularMax > 100.0) {
            specular *= 100.0 / specularMax;
        }
        
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * baseColor;
        
        // Accumulate this light's contribution
        vec3 radiance = light.color.rgb * light.color.a; // intensity in alpha
        Lo += (diffuse + specular) * radiance * NdotL * attenuation;
    }
    
    // ============================================================================
    // ============================================================================
    // RAGE AMBIENT + GLOBAL DIRECTIONAL
    // ============================================================================

    // Layer 1: Hemisphere Ambient (RAGE)
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
    vec3 color = edge_denoise(ambient + directional + Lo + emissive, fragWorldPos);

    // Debug Path Visualization - Only compiled when debug_visualization feature is enabled
    #ifdef DEBUG_VISUALIZATION
    if (push.debug_visualization_enabled == 1) {
        if (push.debug_path == 1) { // GPU-Driven
            color = mix(color, vec3(0.0, 0.0, 1.0), 0.3); // Blue tint
        } else if (push.debug_path == 2) { // Legacy
            color = mix(color, vec3(0.0, 1.0, 0.0), 0.3); // Green tint
        }
    }
    #endif
    
    outColor = vec4(color, 1.0);
    outNormal = vec4(normal, 1.0);
    outAlbedo = vec4(baseColor, 1.0);
    outMotion = motionVector;
}
