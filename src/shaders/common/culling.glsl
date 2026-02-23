// shaders/common/culling.glsl
// Unified Frustum Culling Utilities (Lead GPU Architect SSOT)

void extractFrustumPlanes(mat4 vp, out vec4 planes[6]) {
    vec4 r0 = vec4(vp[0].x, vp[1].x, vp[2].x, vp[3].x);
    vec4 r1 = vec4(vp[0].y, vp[1].y, vp[2].y, vp[3].y);
    vec4 r2 = vec4(vp[0].z, vp[1].z, vp[2].z, vp[3].z);
    vec4 r3 = vec4(vp[0].w, vp[1].w, vp[2].w, vp[3].w);

    planes[0] = r3 + r0; // Left
    planes[1] = r3 - r0; // Right
    planes[2] = r3 + r1; // Bottom
    planes[3] = r3 - r1; // Top
    planes[4] = r2;      // Near (Note: Assumes OpenGL-style NDC or proper Reverse-Z VP)
    planes[5] = r3 - r2; // Far
    
    for (int i = 0; i < 6; i++) {
        float len = length(planes[i].xyz);
        if (len > 0.0001) {
            planes[i] /= len;
        }
    }
}

bool frustumCullSphere(vec3 center, float radius, vec4 planes[6]) {
    for (int i = 0; i < 6; i++) {
        if (dot(planes[i].xyz, center) + planes[i].w < -radius) {
            return true; // Culled
        }
    }
    return false;
}

bool frustumCullAABB(vec3 center, vec3 extents, vec4 planes[6]) {
    for (int i = 0; i < 6; i++) {
        vec3 normal = planes[i].xyz;
        float dist = planes[i].w;
        vec3 positiveVertex = center + extents * sign(normal);
        if (dot(normal, positiveVertex) + dist < 0.0) {
            return true; // Culled
        }
    }
    return false;
}
