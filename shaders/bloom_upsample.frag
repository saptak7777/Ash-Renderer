#version 450

// Bloom upsample pass - tent filter with additive blend

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sourceTexture;

layout(push_constant) uniform PushConstants {
    vec2 texelSize;
    float intensity;
    float _unused;
} pc;

void main() {
    vec2 uv = fragTexCoord;
    vec2 d = pc.texelSize * 0.5; // Half texel for tent filter
    
    // 9-tap tent filter for smooth upsampling
    vec3 s0 = texture(sourceTexture, uv + vec2(-d.x * 2.0, 0.0)).rgb;
    vec3 s1 = texture(sourceTexture, uv + vec2(-d.x, -d.y)).rgb;
    vec3 s2 = texture(sourceTexture, uv + vec2(0.0, -d.y * 2.0)).rgb;
    vec3 s3 = texture(sourceTexture, uv + vec2(d.x, -d.y)).rgb;
    vec3 s4 = texture(sourceTexture, uv + vec2(d.x * 2.0, 0.0)).rgb;
    vec3 s5 = texture(sourceTexture, uv + vec2(d.x, d.y)).rgb;
    vec3 s6 = texture(sourceTexture, uv + vec2(0.0, d.y * 2.0)).rgb;
    vec3 s7 = texture(sourceTexture, uv + vec2(-d.x, d.y)).rgb;
    vec3 s8 = texture(sourceTexture, uv).rgb;
    
    // Tent filter weights
    vec3 result = s8 * 4.0;
    result += (s1 + s3 + s5 + s7) * 2.0;
    result += (s0 + s2 + s4 + s6);
    result /= 16.0;
    
    // Apply intensity
    result *= pc.intensity;
    
    outColor = vec4(result, 1.0);
}
