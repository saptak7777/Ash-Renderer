#version 450

// Bloom threshold pass - extract bright pixels

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrBuffer;

layout(push_constant) uniform PushConstants {
    float exposure;
    float gamma;
    float bloomIntensity;
    float threshold;
} pc;

// Soft threshold to avoid harsh cutoff
vec3 softThreshold(vec3 color, float threshold, float softKnee) {
    float brightness = max(color.r, max(color.g, color.b));
    float soft = brightness - threshold + softKnee;
    soft = clamp(soft, 0.0, 2.0 * softKnee);
    soft = soft * soft / (4.0 * softKnee + 0.00001);
    float contribution = max(soft, brightness - threshold);
    contribution /= max(brightness, 0.00001);
    return color * contribution;
}

void main() {
    vec3 hdr = texture(hdrBuffer, fragTexCoord).rgb;
    
    // Apply soft threshold
    vec3 bright = softThreshold(hdr, pc.threshold, 0.5);
    
    outColor = vec4(bright, 1.0);
}
