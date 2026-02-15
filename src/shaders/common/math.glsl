//! Compute Shader Utilities
//!
//! Common compute shader utilities for GPU-accelerated operations.
//! These provide building blocks for culling, sorting, and parallel algorithms.

#version 450

// ============================================================================
// PARALLEL REDUCTION
// ============================================================================

/// Workgroup-local parallel reduction for finding min/max
/// Usage: Call from compute shader main with local invocation ID

// Shared memory for reduction
shared float sharedData[256];

/// Parallel reduction to find minimum value
/// @param localId - Local invocation ID (0-255)
/// @param value - Input value for this thread
/// @return Minimum value across workgroup (only valid in thread 0)
float parallelMin(uint localId, float value) {
    sharedData[localId] = value;
    barrier();
    
    // Reduction tree
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedData[localId] = min(sharedData[localId], sharedData[localId + stride]);
        }
        barrier();
    }
    
    return sharedData[0];
}

/// Parallel reduction to find maximum value
float parallelMax(uint localId, float value) {
    sharedData[localId] = value;
    barrier();
    
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedData[localId] = max(sharedData[localId], sharedData[localId + stride]);
        }
        barrier();
    }
    
    return sharedData[0];
}

/// Parallel reduction to sum values
float parallelSum(uint localId, float value) {
    sharedData[localId] = value;
    barrier();
    
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sharedData[localId] += sharedData[localId + stride];
        }
        barrier();
    }
    
    return sharedData[0];
}

// ============================================================================
// PREFIX SUM (EXCLUSIVE SCAN)
// ============================================================================

shared uint sharedUint[256];

/// Exclusive prefix sum for 256 elements
/// @param localId - Local invocation ID
/// @param value - Input value
/// @return Exclusive prefix sum (sum of all elements before this one)
uint exclusivePrefixSum(uint localId, uint value) {
    sharedUint[localId] = value;
    barrier();
    
    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < 256; stride <<= 1) {
        uint index = (localId + 1) * stride * 2 - 1;
        if (index < 256) {
            sharedUint[index] += sharedUint[index - stride];
        }
        barrier();
    }
    
    // Clear last element for exclusive scan
    if (localId == 0) {
        sharedUint[255] = 0;
    }
    barrier();
    
    // Down-sweep phase
    for (uint stride = 128; stride > 0; stride >>= 1) {
        uint index = (localId + 1) * stride * 2 - 1;
        if (index < 256) {
            uint temp = sharedUint[index - stride];
            sharedUint[index - stride] = sharedUint[index];
            sharedUint[index] += temp;
        }
        barrier();
    }
    
    return sharedUint[localId];
}

// ============================================================================
// FRUSTUM CULLING HELPERS
// ============================================================================

/// Plane representation (ax + by + cz + d = 0)
struct Plane {
    vec3 normal;
    float distance;
};

/// Extract a plane from a combined view-projection matrix
/// @param mat - Combined VP matrix (column-major)
/// @param face - 0=left, 1=right, 2=bottom, 3=top, 4=near, 5=far
Plane extractPlane(mat4 mat, int face) {
    Plane plane;
    vec4 row;
    
    switch (face) {
        case 0: // Left
            row = mat[3] + mat[0];
            break;
        case 1: // Right
            row = mat[3] - mat[0];
            break;
        case 2: // Bottom
            row = mat[3] + mat[1];
            break;
        case 3: // Top
            row = mat[3] - mat[1];
            break;
        case 4: // Near
            row = mat[3] + mat[2];
            break;
        case 5: // Far
            row = mat[3] - mat[2];
            break;
        default:
            row = vec4(0, 0, 0, 1);
    }
    
    // Normalize plane
    float len = length(row.xyz);
    plane.normal = row.xyz / len;
    plane.distance = row.w / len;
    
    return plane;
}

/// Test if a sphere is outside a plane
/// @return true if sphere is completely outside (culled)
bool sphereOutsidePlane(vec3 center, float radius, Plane plane) {
    float dist = dot(plane.normal, center) + plane.distance;
    return dist < -radius;
}

/// Test if a sphere is inside a frustum (6 planes)
/// @return true if sphere is potentially visible
bool sphereInFrustum(vec3 center, float radius, Plane planes[6]) {
    for (int i = 0; i < 6; i++) {
        if (sphereOutsidePlane(center, radius, planes[i])) {
            return false;
        }
    }
    return true;
}

/// Test if an AABB is outside a plane
bool aabbOutsidePlane(vec3 minBound, vec3 maxBound, Plane plane) {
    // Find the corner most aligned with plane normal
    vec3 pVertex = mix(minBound, maxBound, step(vec3(0), plane.normal));
    return dot(plane.normal, pVertex) + plane.distance < 0.0;
}

/// Test if an AABB is inside a frustum
bool aabbInFrustum(vec3 minBound, vec3 maxBound, Plane planes[6]) {
    for (int i = 0; i < 6; i++) {
        if (aabbOutsidePlane(minBound, maxBound, planes[i])) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// DEPTH HELPERS
// ============================================================================

/// Linearize depth (for perspective projection)
/// @param depth - Non-linear depth from depth buffer (0-1)
/// @param near - Camera near plane
/// @param far - Camera far plane
float linearizeDepth(float depth, float near, float far) {
    return near * far / (far + depth * (near - far));
}

/// Convert linear depth to view-space Z
float depthToViewZ(float depth, float near, float far) {
    return -linearizeDepth(depth, near, far);
}

/// Screen UV to view-space position
vec3 uvDepthToView(vec2 uv, float depth, mat4 invProj) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = invProj * clip;
    return view.xyz / view.w;
}
