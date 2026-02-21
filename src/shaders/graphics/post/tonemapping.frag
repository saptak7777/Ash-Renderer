#version 450

// =============================================================================
// AgX Tonemapping — HDR10 PQ Output Only
//
// Pipeline: Linear Scene Light (R16G16B16A16_SFLOAT)
//           → Exposure
//           → AgX Tone Mapping (optional)
//           → PQ (ST2084) OETF → A2B10G10R10_UNORM_PACK32 HDR swapchain
//
// SDR fallback has been removed. This engine targets 10-bit HDR exclusively.
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
    uint  is_hdr;
} pc;

// =============================================================================
// AgX — Tone Mapping
// Based on the analytical AgX implementation by Troy Sobotka.
// Reference: https://github.com/sobotka/AgX
// =============================================================================

// Linear sRGB/Rec.709 → AgX Log (inset/compressed working space).
// This matrix routes energy away from the visible spectrum boundary so
// the contrast curve never clips.
const mat3 AgXInputMatrix = mat3(
    0.842479062253094,  0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
);

// Inverse: AgX Log → Linear sRGB/Rec.709.
const mat3 AgXOutputMatrix = mat3(
     1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368,  1.15190312990417,   -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116
);

// Apply the AgX input matrix and remap to a log-encoded working range.
vec3 agxLog(vec3 val) {
    // Clamp to a working range that avoids NaN / Inf from negative values.
    val = max(val, 1e-10);
    // Input compression matrix
    val = AgXInputMatrix * val;
    // Map to log2 domain [−10, +6.5 EV] → [0, 1].
    val = clamp(log2(val) / 16.0 + 0.6535, 0.0, 1.0);
    return val;
}

// Analytical sigmoid contrast curve (default look — "base contrast").
// Approximates the sensitometric S-curve of the AgX look.
float agxContrastCurve(float x) {
    float x2 = x * x;
    float x4 = x2 * x2;
    return x
        + x  * (x  - 1.0) * x2 * (x2 - 1.0) * 0.8
        + x2 * (x2 - 1.0) * x4 * 0.2;
}

// Full AgX tone map: input and output are linear sRGB/Rec.709.
vec3 agx(vec3 linearSRGB) {
    // 1. Compress into AgX log working space.
    vec3 encoded = agxLog(linearSRGB);

    // 2. Apply per-channel sigmoid contrast curve.
    encoded.r = agxContrastCurve(encoded.r);
    encoded.g = agxContrastCurve(encoded.g);
    encoded.b = agxContrastCurve(encoded.b);

    // 3. Decompress back to linear sRGB.
    //    The output matrix inverts the AgX input compression.
    vec3 linearOut = AgXOutputMatrix * encoded;

    // 4. Clamp to [0, 1] — the tone mapper guarantees this.
    return clamp(linearOut, 0.0, 1.0);
}

// =============================================================================
// PQ (ST2084) OETF — for A2B10G10R10_UNORM_PACK32 HDR swapchain.
//
// Reference: ITU-R BT.2100, Table 4.
// Input:  linear light, normalised so 1.0 = 10 000 cd/m².
//         (AgX output ∈ [0, 1] maps to [0, 10 000 nits.)
// Output: PQ-encoded value ∈ [0, 1].
// =============================================================================

vec3 pqOetf(vec3 linearNits) {
    // PQ constants (SMPTE ST 2084)
    const float m1 = 0.1593017578125;   //  2610 / 4096 / 4
    const float m2 = 78.84375;          // 2523 / 4096 * 128
    const float c1 = 0.8359375;         //  107 / 128
    const float c2 = 18.8515625;        // 2413 / 128
    const float c3 = 18.6875;           // 2392 / 128

    // Clamp to a valid range to prevent NaN from pow().
    vec3 y = max(linearNits, 0.0);

    vec3 ym1 = pow(y, vec3(m1));
    vec3 num = c1 + c2 * ym1;
    vec3 den = 1.0 + c3 * ym1;
    return pow(num / den, vec3(m2));
}

// =============================================================================
// sRGB OETF — for B8G8R8A8_SRGB swapchain.
// Equivalent to the IEC 61966-2-1 piecewise linearisation.
// Using the simpler power-law approximation here is acceptable because the
// driver's sRGB format already applies the exact transfer function on write.
// However, since we're in a plain UNORM render pass we must apply it manually.
// =============================================================================

float srgbChannel(float linear) {
    if (linear <= 0.0031308)
        return linear * 12.92;
    return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

vec3 srgbOetf(vec3 linear) {
    return vec3(
        srgbChannel(linear.r),
        srgbChannel(linear.g),
        srgbChannel(linear.b)
    );
}


// =============================================================================
// Main
// =============================================================================

void main() {
    // ── 1. Sample buffers ────────────────────────────────────────────────────
    vec3 hdr  = texture(hdrBuffer,  fragTexCoord).rgb;
    vec3 bloom = texture(bloomBuffer, fragTexCoord).rgb;

    // ── 2. Composite bloom ───────────────────────────────────────────────────
    hdr += bloom * pc.bloom_intensity;

    // ── 3. Exposure ──────────────────────────────────────────────────────────
    hdr *= pc.exposure;

    // ── 4. Tone mapping ──────────────────────────────────────────────────────
    vec3 color = hdr;
    if (pc.tonemapper_type > 0) {
        color = agx(hdr);
    }

    // ── 5. Output encoding ───────────────────────────────────────────────────
    // [TEMPORARY_SDR_FALLBACK]: Red Square of Shame logic.
    if (pc.is_hdr == 1u) {
        // HDR path: encode for A2B10G10R10_UNORM_PACK32 + HDR10_ST2084_EXT.
        color = pqOetf(color);
    } else {
        // [DEV NOTICE] SDR Fallback Shame Watermark
        // Draws a 5x5 pure red square in the top-left corner
        if (gl_FragCoord.x < 5.0 && gl_FragCoord.y < 5.0) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        
        // As a temporary fallback for monitors without HDR, we apply the SRGB OETF.
        // The swapchain B8G8R8A8_SRGB format inherently expects sRGB pixels. 
        // Note: Due to the known "Double Gamma" washout, this is severely 
        // compromised. Delete when MSI monitor arrives.
        // color = srgbOetf(color); // Omitted to prevent Double-Gamma.
    }

    outColor = vec4(color, 1.0);
}
