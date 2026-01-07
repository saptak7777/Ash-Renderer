#version 450

// Tonemapping fragment shader
// Applies ACES filmic tonemapping and gamma correction

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrBuffer;
layout(set = 0, binding = 1) uniform sampler2D bloomBuffer;
layout(set = 0, binding = 2) uniform sampler2D ssgiBuffer;

layout(push_constant) uniform PushConstants {
    float exposure;
    float gamma;
    float bloomIntensity;
    float tonemappingEnabled;
} pc;

// ACES filmic tonemapping curve
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // Sample HDR buffer
    vec3 hdr = texture(hdrBuffer, fragTexCoord).rgb;
    
    // Sample Bloom buffer
    vec3 bloom = texture(bloomBuffer, fragTexCoord).rgb;
    
    // Sample SSGI buffer
    vec3 ssgiColor = texture(ssgiBuffer, fragTexCoord).rgb;
    
    // Add Bloom and SSGI
    hdr += bloom * pc.bloomIntensity;
    hdr += ssgiColor; 
    
    // Apply exposure
    hdr *= pc.exposure;
    
    // Apply tonemapping if enabled
    vec3 color = hdr;
    if (pc.tonemappingEnabled > 0.5) {
        color = aces(hdr);
    }
    
    // Apply gamma correction
    color = pow(color, vec3(1.0 / pc.gamma));
    
    outColor = vec4(color, 1.0);
}
