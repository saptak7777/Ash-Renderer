#version 450
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) in vec4 fragPosLightSpace;
layout(location = 5) in vec4 fragTangent;
layout(location = 8) flat in uint fragMaterialIndex;

// G-Buffer Alignment (Outputs)
layout(location = 0) out vec4 outColor;   // HDR Lighting
layout(location = 1) out vec4 outNormal;  // World Space Normal
layout(location = 2) out vec4 outAlbedo;  // Base Color (Albedo)
layout(location = 3) out vec4 outMotion;  // Motion Vectors

// --- Struct Definitions ---

struct MaterialUniform {
    vec4 base_color_factor;
    vec4 emissive_factor;
    vec4 parameters;       // x: metallic, y: roughness, z: occlusion, w: normal_scale
    ivec4 texture_indices; // x: color, y: normal, z: mr, w: occlusion
    int emissive_texture_index;
    int tint_index;
    float alpha_cutoff;
    float _padding;
};

struct HemisphereAmbient {
    vec4 sky;             // .w = intensity
    vec4 ground;          // .w = unused
};

struct DirectionalLight {
    vec4 direction;       // .w = shadow enabled
    vec4 color_intensity; // .rgb = color, .w = intensity
};

struct SceneLighting {
    HemisphereAmbient ambient;
    DirectionalLight directional;
    uint point_light_count;
    uint num_tiles_x;
    uint num_tiles_y;
    uint tile_size;
    int ibl_irradiance_index;
    int ibl_prefilter_index;
    int ibl_brdf_lut_index;
    float ibl_intensity;
};

struct MvpMatrices {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 prev_view_proj;
    mat4 light_space_matrix;
    mat4 normal_matrix;
    vec4 camera_pos;
    SceneLighting scene_lighting;
};

struct Light {
    vec4 position;   // xyz = position, w = radius
    vec4 color;      // rgb = color, a = intensity
    vec4 direction;  // xyz = direction, w = type
    vec4 params;     // w = enabled
};

// --- Buffer References (BDA) ---

layout(buffer_reference, std430) readonly buffer SceneData {
    MvpMatrices matrices;
};

layout(buffer_reference, std430) readonly buffer MaterialStorage {
    MaterialUniform materials[];
};

layout(buffer_reference, std430) readonly buffer LightBuffer {
    Light lights[];
};

layout(buffer_reference, std430) readonly buffer GlobalBuffers {
    vec4 data[];
};

// --- Bindings ---

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 4) readonly buffer BindlessBuffers {
    vec4 data[];
} bindless_buffers[];

layout(push_constant) uniform DrawPushConstants {
    uint64_t frame_ptr;
    uint64_t vertex_ptr;
    uint64_t instance_ptr;
    uint64_t material_ptr;
    uint64_t index_ptr;
    uint64_t light_ptr;
    uint64_t tile_ptr;
    
    layout(offset = 56) uint vsm_page_index;
    uint vsm_cache_index;
    
    layout(offset = 64) uint64_t transform_ptr;
    uint transform_index;
    uint material_index;
    uint use_instancing;
} push;

const float PI = 3.14159265359;

// --- Cook-Torrance PBR Functions ---

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.000001);
}

float geometry_schlick_ggx(float NdotV, float k) {
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return geometry_schlick_ggx(NdotV, k) * geometry_schlick_ggx(NdotL, k);
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 calculatePBR(vec3 L, vec3 V, vec3 N, vec3 F0, vec3 baseColor, float metallic, float roughness, vec3 lightColor) {
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotL = max(dot(N, L), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float D = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);
    vec3 F = fresnel_schlick(VdotH, F0);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;
    
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;

    return (diffuse + specular) * lightColor * NdotL;
}

void main() {
    // 1. Fetch Pointers and Material
    SceneData scene = SceneData(push.frame_ptr);
    MvpMatrices mvp = scene.matrices;
    
    MaterialStorage material_storage = MaterialStorage(push.material_ptr);
    uint matID = (push.use_instancing == 1) ? fragMaterialIndex : push.material_index;
    MaterialUniform mat = material_storage.materials[nonuniformEXT(matID)];

    // 2. Base Vectors
    vec3 N_geom = normalize(fragNormal);
    vec3 V = normalize(mvp.camera_pos.xyz - fragWorldPos);

    // 3. Material Properties
    vec4 baseSample = mat.texture_indices.x >= 0 
        ? texture(textures[nonuniformEXT(mat.texture_indices.x)], fragUV) 
        : vec4(1.0);
    vec3 baseColor = baseSample.rgb * mat.base_color_factor.rgb;
    float alpha = baseSample.a * mat.base_color_factor.a;

    if (mat.tint_index >= 0) {
        baseColor *= bindless_buffers[nonuniformEXT(mat.tint_index)].data[0].rgb;
    }

    if (alpha < mat.alpha_cutoff) {
        discard;
    }

    // Normal Mapping
    vec3 T_raw = fragTangent.xyz;
    vec3 T = length(T_raw) > 0.001 ? normalize(T_raw) : vec3(1.0, 0.0, 0.0);
    T = normalize(T - dot(T, N_geom) * N_geom);
    
    vec3 N = N_geom;
    if (!gl_FrontFacing) {
        N = -N;
        T = -T;
    }
    
    vec3 B = cross(N, T) * fragTangent.w;
    mat3 TBN = mat3(T, B, N);
    
    if (mat.texture_indices.y >= 0) {
        vec3 mapSample = texture(textures[nonuniformEXT(mat.texture_indices.y)], fragUV).xyz;
        if (length(mapSample) > 0.001) {
            vec3 mapNormal = mapSample * 2.0 - 1.0;
            mapNormal.xy *= mat.parameters.w; // normal_scale
            N = normalize(TBN * mapNormal);
        }
    }

    float metallic = mat.parameters.x;
    float roughness = clamp(mat.parameters.y, 0.05, 1.0);
    
    if (mat.texture_indices.z >= 0) {
        vec4 mrSample = texture(textures[nonuniformEXT(mat.texture_indices.z)], fragUV);
        metallic *= mrSample.b;
        roughness = clamp(roughness * mrSample.g, 0.05, 1.0);
    }

    float occlusion = 1.0;
    if (mat.texture_indices.w >= 0) {
        occlusion = mix(1.0, texture(textures[nonuniformEXT(mat.texture_indices.w)], fragUV).r, mat.parameters.z);
    }

    // 4. Lighting Calculation
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    vec3 totalLo = vec3(0.0);

    // Directional Light
    {
        vec3 L = normalize(-mvp.scene_lighting.directional.direction.xyz);
        vec3 lightColor = mvp.scene_lighting.directional.color_intensity.rgb * mvp.scene_lighting.directional.color_intensity.w;
        totalLo += calculatePBR(L, V, N, F0, baseColor, metallic, roughness, lightColor);
    }

    // Dynamic Point Lights
    if (push.light_ptr != 0) {
        LightBuffer lb = LightBuffer(push.light_ptr);
        for (uint i = 0; i < mvp.scene_lighting.point_light_count; i++) {
            Light light = lb.lights[i];
            vec3 L = normalize(light.position.xyz - fragWorldPos);
            float dist = length(light.position.xyz - fragWorldPos);
            
            float attenuation = 1.0 / (dist * dist + 1.0);
            float radiusAtten = clamp(1.0 - (dist / light.position.w), 0.0, 1.0);
            vec3 lightColor = light.color.rgb * light.color.a * attenuation * radiusAtten;
            
            totalLo += calculatePBR(L, V, N, F0, baseColor, metallic, roughness, lightColor);
        }
    }

    // 5. Ambient
    float h = N.y * 0.5 + 0.5;
    vec3 ambient = mix(mvp.scene_lighting.ambient.ground.rgb, mvp.scene_lighting.ambient.sky.rgb, h) * mvp.scene_lighting.ambient.sky.w;
    ambient *= baseColor * occlusion;

    vec3 emissive = mat.emissive_factor.rgb;
    if (mat.emissive_texture_index >= 0) {
        emissive *= texture(textures[nonuniformEXT(mat.emissive_texture_index)], fragUV).rgb;
    }

    // Final Resolve
    vec3 finalColor = ambient + totalLo + emissive;

    // G-Buffer Output
    outColor = vec4(finalColor, 1.0);
    outNormal = vec4(N * 0.5 + 0.5, 1.0);
    outAlbedo = vec4(baseColor, alpha);
    outMotion = vec4(0.0); // Placeholder for motion vectors
}
