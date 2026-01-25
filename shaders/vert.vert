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

void main() {
    // Access Frame Data via BDA
    FrameData frame = FrameData(push.frame_ptr);
    
    // BDA Vertex Pulling: Load vertex data from global vertex heap
    VertexBuffer vertex = load_vertex(push.vertex_ptr, gl_VertexIndex);
    
    vec3 inPosition = vertex.position;
    vec3 inNormal = vertex.normal;
    vec2 inUV = vertex.uv;
    vec3 inColor = vertex.color;
    vec4 inTangent = vertex.tangent;

    mat4 modelMatrix;
    if (push.use_instancing != 0) {
        InstanceBuffer instance_buffer = InstanceBuffer(push.instance_ptr);
        modelMatrix = instance_buffer.instances[gl_InstanceIndex].model;
    } else {
        modelMatrix = push.model;
    }

    vec4 worldPosition = modelMatrix * vec4(inPosition, 1.0);
    gl_Position = frame.view_proj * worldPosition;

    fragColor = inColor;
    if (push.use_instancing != 0) {
        InstanceBuffer instance_buffer = InstanceBuffer(push.instance_ptr);
        fragColor *= instance_buffer.instances[gl_InstanceIndex].color.rgb;
    }
    fragUV = inUV;
    
    mat3 normalMat;
    if (push.use_instancing != 0) {
        // PER-INSTANCE NORMAL MATRIX: Calculate from instance model matrix to fix lighting on rotated objects
        normalMat = transpose(inverse(mat3(modelMatrix)));
    } else {
        normalMat = mat3(frame.normal_matrix);
    }
    
    fragNormal = normalize(normalMat * inNormal);
    fragTangent = vec4(normalize(normalMat * inTangent.xyz), inTangent.w);
    
    fragWorldPos = worldPosition.xyz;
    fragPosLightSpace = frame.light_space_matrix * worldPosition;

    vec4 currentClip = frame.view_proj * worldPosition;
    vec4 prevClip = frame.prev_view_proj * worldPosition;
    motionVector = (currentClip.xy / currentClip.w - prevClip.xy / prevClip.w) * 0.5;
}
