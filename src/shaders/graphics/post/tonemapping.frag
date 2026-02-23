#version 450

// =============================================================================
// Lean Display Architecture — AgX + SDR Output
//
// Pipeline: Linear Scene Light (B10G11R11_UFLOAT) 
//           → Exposure
//           → AgX Tone Mapping (SDR Optimized)
//           → Gamma 2.2 OETF
//           → IGN TPDF Dither → 8-bit SDR swapchain
// =============================================================================

layout(location = 0) in  vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrBuffer;
layout(set = 0, binding = 1) uniform sampler2D bloomBuffer;
layout(set = 0, binding = 2) uniform sampler2D ssgiBuffer;

// Must match PostProcessPushConstants in fullscreen.rs exactly (16 bytes).
layout(push_constant) uniform PushConstants {
    float exposure;
    float bloom_intensity;
    uint  tonemapper_type;
    float gamma; // Display calibration target.
} pc;

// =============================================================================
// AgX — Tone Mapping
// Based on the analytical AgX implementation by Troy Sobotka.
// =============================================================================

const mat3 AgXInputMatrix = mat3(
    0.842479062253094,  0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
);

const mat3 AgXOutputMatrix = mat3(
     1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368,  1.15190312990417,   -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116
);

vec3 agxLog(vec3 val) {
    val = max(val, 1e-10);
    val = AgXInputMatrix * val;
    val = clamp(log2(val) / 16.0 + 0.6535, 0.0, 1.0);
    return val;
}

float agxContrastCurve(float x) {
    float x2 = x * x;
    float x4 = x2 * x2;
    return x
        + x  * (x  - 1.0) * x2 * (x2 - 1.0) * 0.8
        + x2 * (x2 - 1.0) * x4 * 0.2;
}

vec3 agx(vec3 linearSRGB) {
    vec3 encoded = agxLog(linearSRGB);
    encoded.r = agxContrastCurve(encoded.r);
    encoded.g = agxContrastCurve(encoded.g);
    encoded.b = agxContrastCurve(encoded.b);
    return clamp(AgXOutputMatrix * encoded, 0.0, 1.0);
}

// =============================================================================
// Interleaved Gradient Noise (IGN) TPDF Dither
// Reference: Jorge Jimenez (Activision) - "Next Generation Post Processing"
// =============================================================================

float ign(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

vec3 dither(vec3 color, vec2 pixel) {
    // Generate TPDF (Triangular Probability Density Function) noise
    // By noise = (noise1 + noise2) - 0.5
    float noise = ign(pixel);
    // 8-bit quantization step
    float lsb = 1.0 / 255.0;
    return color + (noise - 0.5) * lsb;
}

// =============================================================================
// Main
// =============================================================================

void main() {
    // 1. Sample buffers (Linear B10G11R11_UFLOAT)
    vec3 hdr = texture(hdrBuffer, fragTexCoord).rgb;
    vec3 bloom = texture(bloomBuffer, fragTexCoord).rgb;

    // 2. Composite
    hdr += bloom * pc.bloom_intensity;
    hdr *= pc.exposure;

    // 3. Tone Mapping (HDR -> SDR [0, 1])
    vec3 color = hdr;
    if ((pc.tonemapper_type & 1) != 0) {
        color = agx(hdr);
    }

    // 4. Gamma 2.2 OETF (Standard SDR display curve)
    color = pow(color, vec3(1.0 / pc.gamma));

    // 5. IGN Dither (Prevents 8-bit banding)
    color = dither(color, gl_FragCoord.xy);

    outColor = vec4(color, 1.0);
}
