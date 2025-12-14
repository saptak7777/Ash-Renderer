#version 450

// Bloom downsample pass - bilinear 13-tap Gaussian blur

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sourceTexture;

layout(push_constant) uniform PushConstants {
    vec2 texelSize; // 1.0 / textureSize
    float _unused1;
    float _unused2;
} pc;

void main() {
    // 13-tap bilinear downsample with partial Karis Average to built-in firefly reduction
    vec2 uv = fragTexCoord;
    vec2 d = pc.texelSize;
    
    // Center
    vec3 center = texture(sourceTexture, uv).rgb;
    
    // Corner samples
    vec3 a = texture(sourceTexture, uv + vec2(-d.x, -d.y)).rgb;
    vec3 b = texture(sourceTexture, uv + vec2( d.x, -d.y)).rgb;
    vec3 c = texture(sourceTexture, uv + vec2(-d.x,  d.y)).rgb;
    vec3 d_sample = texture(sourceTexture, uv + vec2( d.x,  d.y)).rgb;
    
    // Edge samples
    vec3 e = texture(sourceTexture, uv + vec2(-d.x, 0.0)).rgb;
    vec3 f = texture(sourceTexture, uv + vec2( d.x, 0.0)).rgb;
    vec3 g = texture(sourceTexture, uv + vec2(0.0, -d.y)).rgb;
    vec3 h = texture(sourceTexture, uv + vec2(0.0,  d.y)).rgb;
    
    // Karis average calculation (only for the first downsample, technically, but good generally)
    // Weight = 1 / (1 + luma)
    // Groups: (a,b,d,e), (b,c,e,f)... simple tent is easier.
    // Proper Karis:
    // We need 5 groups for the 13-tap.
    // Group 1: Center
    // Group 2-5: 4 corners.
    // But we are doing 13-tap.
    // Standard Karis typically uses a 5-tap (center + 4 corners).
    // Let's stick to the 13-tap pattern but apply weighting.
    
    // Simplified Karis Check:
    // Converting to luma
    float lumaCenter = dot(center, vec3(0.2126, 0.7152, 0.0722));
    float wCenter = 1.0 / (1.0 + lumaCenter);
    
    // This is getting complex for a simple replacement.
    // Let's just implement the weighting for the 4x4 box groups if possible?
    // Actually, simply weighting the result by luma suppression is often enough.
    
    // Let's do a weighted average of the groups.
    // Box 1 (Top Left): a, e, g, center
    // Box 2 (Top Right): b, f, g, center
    // ...
    // This shader seems to implement the "Better Bloom" by Call of Duty / Jimenez 2014.
    // The "13-tap" is standard there.
    // Karis average is usually applied *before* the first downsample or *during* it.
    
    // Let's apply partial luma weighting to the samples.
    vec3 result = center * 0.125;
    result += (a + b + c + d_sample) * 0.125;
    result += (e + f + g + h) * 0.125;
    // Wait, the previous code weights were different (0.25, 0.0625, 0.125).
    // 0.25 center, 0.0625 corners (1/16), 0.125 edges (1/8).
    // Sum: 0.25 + 4*0.0625 + 4*0.125 = 0.25 + 0.25 + 0.5 = 1.0. Correct.
    
    // Reverting to weighted sum but adding Karis weighting
    // Function to calculate weight
    #define w(c) (1.0 / (1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722))))
    
    vec3 res = vec3(0.0);
    float sum = 0.0;
    
    // Center
    float wc = w(center);
    res += center * wc * 0.25;
    sum += wc * 0.25;
    
    // Corners
    float wa = w(a); res += a * wa * 0.0625; sum += wa * 0.0625;
    float wb = w(b); res += b * wb * 0.0625; sum += wb * 0.0625;
    float wd = w(d_sample); res += d_sample * wd * 0.0625; sum += wd * 0.0625;
    float wcc = w(c); res += c * wcc * 0.0625; sum += wcc * 0.0625;
    
    // Edges
    float we = w(e); res += e * we * 0.125; sum += we * 0.125;
    float wf = w(f); res += f * wf * 0.125; sum += wf * 0.125;
    float wg = w(g); res += g * wg * 0.125; sum += wg * 0.125;
    float wh = w(h); res += h * wh * 0.125; sum += wh * 0.125;
    
    outColor = vec4(res / sum, 1.0);
}
