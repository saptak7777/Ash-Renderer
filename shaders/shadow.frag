#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// Shadow map fragment shader - outputs variance moments for VSM
// VSM uses a physical cache texture (R32G32_SFLOAT) to store (depth, depth^2)
// This shader works with the dual-attachment VSM pass (color + depth)

layout(location = 0) in vec2 inUV;

// Output variance moments to color attachment 0 (R32G32_SFLOAT)
layout(location = 0) out vec2 outMoments;

layout(push_constant) uniform PushConstants {
    // Skip Vertex push constants (0-159)
    // Fragment push constants start at offset 160
    layout(offset = 160) int base_color_index; 
} pc;

// Binding 0: Textures (Set 1)
layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
    // Alpha testing for transparent materials
    // NOTE: Currently disabled due to push constant layout uncertainty
    // The base_color_index offset needs verification against CPU-side structure
    /*
    if (pc.base_color_index >= 0) {
        float alpha = texture(textures[nonuniformEXT(pc.base_color_index)], inUV).a;
        if (alpha < 0.1) {
            discard;
        }
    }
    */
    
    // Variance Shadow Mapping: Store (depth, depth^2) for statistical filtering
    // Hardware depth testing happens on the D32_SFLOAT depth attachment
    // We output variance moments to the R32G32_SFLOAT color attachment
    float depth = gl_FragCoord.z;
    outMoments = vec2(depth, depth * depth);
}
