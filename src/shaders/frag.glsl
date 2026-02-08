#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) in vec4 fragPosLightSpace;
layout(location = 5) in vec4 fragTangent;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 light_space_matrix;
    mat4 normal_matrix;
    vec4 camera_pos;
    vec4 light_direction;
    vec4 light_color;
    vec4 ambient_color;
} mvp;

// Material properties via push constants
layout(push_constant) uniform Material {
    layout(offset = 128) vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    float occlusion_strength;
    float alpha_cutoff;
    int alpha_mode;
    int base_color_texture_set;
    int normal_texture_set;
    int metallic_roughness_texture_set;
    int occlusion_texture_set;
    int emissive_texture_set;
    int tint_index;
    // Padding to 16-byte align emissive_factor
    vec4 emissive_factor;
} material;

// Bindless texture array
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 2, binding = 0) uniform sampler2D textures[];
layout(set = 2, binding = 2) readonly buffer Tints {
    vec4 colors[];
} tints[];


layout(set = 3, binding = 0) uniform sampler2D shadowMap;

const float PI = 3.14159265359;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // slope-aware bias
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // Default to 1.0 (unshadowed) if the coordinate is beyond the shadow map range
    // but default to 0.0 if the shadow sampler is empty (not possible to check directly,
    // so we rely on correctly cleared maps or specific checks).
    if(projCoords.z > 1.0)
        return 0.0;
    
    // PCF 5x5 using textureGather (4 samples per call = ~8 gathers for 25 samples)
    // textureGather returns 4 texels in a 2x2 quad from the specified corner
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    vec2 uv = projCoords.xy;
    
    float shadow = 0.0;
    
    // Sample 6 gather points to cover 5x5 area with overlap
    // Each gather returns depths from a 2x2 quad
    
    // Row 0-1 (y = -2 to -1)
    vec4 g0 = textureGather(shadowMap, uv + vec2(-2.0, -2.0) * texelSize);
    vec4 g1 = textureGather(shadowMap, uv + vec2( 0.0, -2.0) * texelSize);
    vec4 g2 = textureGather(shadowMap, uv + vec2( 2.0, -2.0) * texelSize);
    
    // Row 2-3 (y = 0 to 1)
    vec4 g3 = textureGather(shadowMap, uv + vec2(-2.0,  0.0) * texelSize);
    vec4 g4 = textureGather(shadowMap, uv + vec2( 0.0,  0.0) * texelSize);
    vec4 g5 = textureGather(shadowMap, uv + vec2( 2.0,  0.0) * texelSize);
    
    // Row 4 (y = 2) - partial coverage
    vec4 g6 = textureGather(shadowMap, uv + vec2(-2.0,  2.0) * texelSize);
    vec4 g7 = textureGather(shadowMap, uv + vec2( 0.0,  2.0) * texelSize);
    vec4 g8 = textureGather(shadowMap, uv + vec2( 2.0,  2.0) * texelSize);
    
    // Compare each gathered depth against currentDepth - bias
    float compareDepth = currentDepth - bias;
    
    // Count samples in shadow from each gather (4 comparisons per gather)
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g0)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g1)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g2)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g3)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g4)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g5)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g6)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g7)), vec4(1.0));
    shadow += dot(vec4(greaterThan(vec4(compareDepth), g8)), vec4(1.0));
    
    // 9 gathers * 4 samples = 36 samples, but 5x5 = 25 samples
    // This provides slightly more blur (6x6 effective) which is acceptable
    // Normalize to approximate 5x5 result
    return shadow / 36.0;
}

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Optimized: Schlick-GGX with fast reciprocal approximation
float geometry_schlick_ggx_fast(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125; // (r*r)/8 = (r*r)*0.125
    return geometry_schlick_ggx_fast(NdotV, k) * geometry_schlick_ggx_fast(NdotL, k);
}

// Optimized: Fresnel-Schlick with spherical gaussian approximation (faster pow)
vec3 fresnel_schlick_fast(float cosTheta, vec3 F0) {
    // Spherical Gaussian approximation: exp2(-5.55473 * x - 6.98316 * x) ≈ (1-x)^5
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    float t2 = t * t;
    float t5 = t2 * t2 * t; // t^5 without pow()
    return F0 + (1.0 - F0) * t5;
}

void main() {
    vec3 lightColor = mvp.light_color.xyz;
    vec3 ambientColor = mvp.ambient_color.xyz;

    vec3 viewDir = normalize(mvp.camera_pos.xyz - fragWorldPos);
    vec3 L = normalize(-mvp.light_direction.xyz);

    // Sample base color
    // Sample base color (bindless)
    vec4 baseSample = material.base_color_texture_set >= 0
        ? texture(textures[nonuniformEXT(material.base_color_texture_set)], fragUV)
        : vec4(1.0);
    vec3 baseColor = baseSample.rgb * material.base_color_factor.rgb;
    float alpha = baseSample.a * material.base_color_factor.a;

    // Apply bindless tint if present
    if (material.tint_index >= 0) {
        baseColor *= tints[nonuniformEXT(material.tint_index)].colors[0].rgb;
    }

    // Alpha testing
    if (alpha < material.alpha_cutoff) {
        discard;
    }

    // Tangent-based Normal Mapping
    vec3 N = normalize(fragNormal);
    vec3 T_raw = fragTangent.xyz;
    vec3 T = length(T_raw) > 0.001 ? normalize(T_raw) : vec3(1.0, 0.0, 0.0); // Safe fallback
    
    // Gram-Schmidt orthogonalization
    T = normalize(T - dot(T, N) * N);
    
    // Flip normal for backfaces to support double-sided rendering correctly
    if (!gl_FrontFacing) {
        N = -N;
        T = -T;
    }
    
    // Bitangent with handedness
    vec3 B = cross(N, T) * fragTangent.w;
    
    mat3 TBN = mat3(T, B, N);
    
    vec3 normal = N;
    if (material.normal_texture_set >= 0) {
        vec3 mapSample = texture(textures[nonuniformEXT(material.normal_texture_set)], fragUV).xyz;
        // Check for validity (e.g. if mipmapping averages to 0)
        if (length(mapSample) > 0.001) {
            vec3 mapNormal = mapSample * 2.0 - 1.0;
            mapNormal.xy *= material.normal_scale;
            // Safe normalize result
            vec3 mapDir = TBN * mapNormal;
            if (length(mapDir) > 0.001) {
                normal = normalize(mapDir);
            }
        }
    }


    // Material parameters
    float metallic = material.metallic_factor;
    float roughness = clamp(material.roughness_factor, 0.05, 1.0); // Min roughness to prevent black pixels
    
    if (material.metallic_roughness_texture_set >= 0) {
        vec4 mrSample = texture(textures[nonuniformEXT(material.metallic_roughness_texture_set)], fragUV);
        metallic = metallic * mrSample.b;
        roughness = clamp(roughness * mrSample.g, 0.05, 1.0);
    }

    // Ambient occlusion (bindless)
    float occlusion = 1.0;
    if (material.occlusion_texture_set >= 0) {
        occlusion = mix(1.0, texture(textures[nonuniformEXT(material.occlusion_texture_set)], fragUV).r, material.occlusion_strength);
    }

    // PBR
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    vec3 halfDir = normalize(viewDir + L);
    float NdotV = max(dot(normal, viewDir), 0.001);
    float NdotL = max(dot(normal, L), 0.0);
    float NdotH = max(dot(normal, halfDir), 0.0);
    float VdotH = max(dot(viewDir, halfDir), 0.0);

    float D = distribution_ggx(NdotH, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);
    vec3 F = fresnel_schlick_fast(VdotH, F0);

    vec3 numerator = D * G * F;
    float denom = 4.0 * NdotV * NdotL + 0.001;
    vec3 specular = numerator / denom;
    
    // Clamp excessive specular to prevent fireflies (Conservative Specular Cap)
    specular = min(specular, vec3(10.0) / max(vec3(0.04), F0));

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;
    
    // Shadow mapping calculations using geometric normal (N) for bias to avoid self-shadowing.
    float shadow = ShadowCalculation(fragPosLightSpace, N, L);

    // Direct lighting with shadow
    vec3 Lo = (diffuse + specular) * lightColor * NdotL * (1.0 - shadow);
    
    // Debug: Specular output
    // outColor = vec4(specular * 2.0, 1.0); return;
    
    // Ambient
    vec3 ambient = ambientColor * baseColor * occlusion;
    
    // Emissive (bindless)
    vec3 emissive = material.emissive_factor.rgb;
    if (material.emissive_texture_set >= 0) {
        emissive *= texture(textures[nonuniformEXT(material.emissive_texture_set)], fragUV).rgb;
    }

    vec3 color = ambient + Lo + emissive;
    
    // Reinhard tonemapping
    color = color / (color + vec3(1.0));

    outColor = vec4(color, 1.0);
}
