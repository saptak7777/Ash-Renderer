#version 450

//! Motion Vector Fragment Shader
//!
//! Calculates per-pixel motion vectors by comparing current and previous
//! frame positions. Output is in UV space for use with TSR/TAA.

layout(location = 0) in vec4 inCurrentPos;
layout(location = 1) in vec4 inPreviousPos;

layout(location = 0) out vec2 outMotion; // RG16F format

void main() {
    // Convert clip-space to NDC (Normalized Device Coordinates)
    vec2 currentNDC = inCurrentPos.xy / inCurrentPos.w;
    vec2 previousNDC = inPreviousPos.xy / inPreviousPos.w;
    
    // Convert NDC [-1, 1] to UV space [0, 1]
    vec2 currentUV = currentNDC * 0.5 + 0.5;
    vec2 previousUV = previousNDC * 0.5 + 0.5;
    
    // Motion vector = current position - previous position
    // This gives the direction and magnitude of motion in UV space
    outMotion = currentUV - previousUV;
}
