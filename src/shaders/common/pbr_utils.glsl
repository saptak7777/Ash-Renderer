#ifndef PBR_UTILS_GLSL
#define PBR_UTILS_GLSL

#define PI 3.14159265359
const float MAX_REFLECTION_LOD = 7.0;

/**
 * Fresnel Schlick with Roughness interaction
 * F0: Base reflectance (0.04 for dielectrics, albedo for metals)
 */
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/**
 * Split-Sum Approximation IBL Contribution
 * NdotV: dot(Normal, View)
 * N: World Normal
 * R: Reflection Vector (reflect(-V, N))
 */
vec3 getIBLContribution(float NdotV, vec3 N, vec3 R, vec3 F0, float roughness, vec3 albedo, float metallic, float occlusion) {
    // 1. Fresnel term for IBL
    vec3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    // 2. Diffuse Part: Irradiance Map
    vec3 irradiance = texture(u_IrradianceMap, N).rgb;
    vec3 diffuse = (irradiance * albedo) / PI;

    // 3. Specular Part: Prefilter Map (LD) + BRDF LUT (DFG)
    vec3 prefilteredColor = textureLod(u_PrefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_BrdfLut, vec2(NdotV, roughness)).rg;
    
    // Combining term: kS * brdf.x + brdf.y
    vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);

    return (kD * diffuse + specular) * occlusion;
}

#endif // PBR_UTILS_GLSL
