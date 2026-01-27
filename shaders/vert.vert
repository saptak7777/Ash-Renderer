#version 450
// BDA Vertex Pulling Implementation
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "include/structures.glsl"
#include "include/vertex_pulling.glsl"

// Output attributes
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) centroid out vec3 fragNormal;
layout(location = 3) sample out vec3 fragWorldPos;
layout(location = 4) out vec4 fragPosLightSpace;
layout(location = 5) out vec4 fragTangent;
layout(location = 6) out vec2 motionVector;
layout(location = 7) flat out uint fragInstanceIndex;

void main() {
    // Access Frame Data via BDA
    FrameData frame = FrameData(push.frame_ptr);
    
    mat4 model = push.model;
    int vertex_offset = 0;
    
    // Modern BDA Instancing
    if (push.use_instancing == 1 && push.instance_ptr != 0) {
        InstanceBuffer instance_ctx = InstanceBuffer(push.instance_ptr);
        // gl_InstanceIndex correctly accounts for firstInstance in indirect draws
        InstanceData instance = instance_ctx.instances[gl_InstanceIndex];
        model = instance.model;
        vertex_offset = instance.vertex_offset;
    }

    // BDA Vertex Pulling: Load vertex data from global vertex heap
    VertexBuffer vertex = load_vertex(push.vertex_ptr, gl_VertexIndex + vertex_offset);
    
    // Transform position to world space
    vec4 worldPosition = model * vec4(vertex.position, 1.0);
    
    // Calculate world-space normal
    vec3 worldNormal = normalize(mat3(model) * vertex.normal);
    
    // Output to clip space
    gl_Position = frame.view_proj * worldPosition;
    
    // Pass through to fragment shader
    fragColor = vertex.color;
    fragUV = vertex.uv;
    fragNormal = worldNormal;
    fragWorldPos = worldPosition.xyz;
    fragTangent = vec4(mat3(push.model) * vertex.tangent.xyz, vertex.tangent.w);
    motionVector = vec2(0.0);
    fragInstanceIndex = gl_InstanceIndex;
}
