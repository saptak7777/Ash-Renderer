#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) in vec4 fragPosLightSpace;
layout(location = 5) in vec4 fragTangent;
layout(location = 6) in vec2 motionVector;

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
    vec4 light_direction;
    vec4 light_color;
    vec4 ambient_color;
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
} material_buffer;

// Tint buffer (still used by some parts, but consolidated to Set 1 if needed - 
// however Renderer doesn't seem to bind separate tint buffers in BindlessManager yet)
// Let's keep it in Set 2 for now IF Renderer still binds it there, 
// but wait, DescriptorManager Set 2 is Environment.
// Tints SHOULD be in Bindless (Set 1) if they are storage buffers.
// For now, I'll rely on MaterialUniform's fields.

layout(push_constant) uniform PushConstants {
    // Vertex stage (0-127)
    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;

    // Fragment stage (128-255)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint debug_path; // 0: None, 1: GPU-Driven, 2: Legacy
    layout(offset = 136) uint _material_padding[2];
} push;

// Set 2: Environment (ShadowMap + IBL)
layout(set = 2, binding = 0) uniform sampler2D shadowMap;

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

void main() {
    MaterialUniform mat = material_buffer.materials[push.material_index];

    vec3 lightColor = mvp.light_color.xyz;
    vec3 ambientColor = mvp.ambient_color.xyz;

    vec3 viewDir = normalize(mvp.camera_pos.xyz - fragWorldPos);
    vec3 lightDir = normalize(-mvp.light_direction.xyz);

    // Sample base color
    int base_color_idx = mat.texture_indices.x;
    vec4 base_color_factor = mat.base_color_factor;

    vec4 baseSample = base_color_idx >= 0
        ? texture(textures[nonuniformEXT(base_color_idx)], fragUV)
        : vec4(1.0);
    // Apply sRGB-to-linear conversion for physically-based color handling
    vec3 baseSampleLinear = srgb_to_linear(baseSample.rgb);
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
    
    if (!gl_FrontFacing) {
        N = -N;
        T = -T;
    }
    
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

    float NdotL = max(dot(normal, lightDir), 0.0);

    // Material parameters
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

    // PBR
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    vec3 halfDir = normalize(viewDir + lightDir);
    float NdotV = max(dot(normal, viewDir), 0.001);
    float NdotH = max(dot(normal, halfDir), 0.0);
    float VdotH = max(dot(viewDir, halfDir), 0.0);

    float D = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);
    vec3 F = fresnel_schlick_fast(VdotH, F0);

    vec3 numerator = D * G * F;
    float denom = 4.0 * NdotV * NdotL + 0.001;
    vec3 specular = numerator / denom;
    
    float specularMax = max(max(specular.r, specular.g), specular.b);
    if (specularMax > 100.0) {
        specular *= 100.0 / specularMax;
    }

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;
    
    float shadow = ShadowCalculation(fragPosLightSpace, N, lightDir);

    vec3 Lo = (diffuse + specular) * lightColor * NdotL * (1.0 - shadow);
    vec3 ambient = ambientColor * baseColor * occlusion;
    
    // Emissive
    int emissive_idx = mat.emissive_texture_index;
    vec3 emissive = mat.emissive_factor.rgb;
    if (emissive_idx >= 0) {
        emissive *= texture(textures[nonuniformEXT(emissive_idx)], fragUV).rgb;
    }

    vec3 color = ambient + Lo + emissive;
    
    // Debug Path Visualization
    if (push.debug_path == 1) { // GPU-Driven
        color = mix(color, vec3(0.0, 0.0, 1.0), 0.3); // Blue tint
    } else if (push.debug_path == 2) { // Legacy
        color = mix(color, vec3(0.0, 1.0, 0.0), 0.3); // Green tint
    }
    
    outColor = vec4(color, 1.0);
    outNormal = vec4(normal, 1.0);
    outAlbedo = vec4(baseColor, 1.0);
    outMotion = motionVector;
}
