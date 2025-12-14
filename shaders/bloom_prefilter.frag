#version 450

// Bloom Pre-Filter Pass
// Applies luminance threshold with soft knee to suppress fireflies BEFORE downsampling.
// This is more effective than weighting during downsample alone.

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sourceTexture;

layout(push_constant) uniform PushConstants {
    vec2 texelSize;     // 1.0 / textureSize
    float threshold;    // Luminance threshold (default: 1.0)
    float softKnee;     // Soft knee factor (default: 0.5)
} pc;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

void main() {
    vec3 color = texture(sourceTexture, fragTexCoord).rgb;
    float luma = dot(color, LUMA);
    
    // Soft thresholding with knee
    // Prevents hard cutoff artifacts at the threshold boundary
    float knee = pc.threshold * pc.softKnee;
    float x = luma - (pc.threshold - knee);
    
    float response;
    if (x <= 0.0) {
        response = 0.0;  // Below threshold - no bloom contribution
    } else if (x >= 2.0 * knee) {
        response = x;    // Above knee - linear response
    } else {
        // Quadratic transition in the knee region
        response = (x * x) / (4.0 * knee + 0.0001);
    }
    
    // Scale output by the response ratio
    // Avoid division by zero for pure black pixels
    float safeRatio = response / max(luma, 0.0001);
    outColor = vec4(color * safeRatio, 1.0);
}
