#version 450
#extension GL_EXT_nonuniform_qualifier : enable




layout(location = 0) in vec2 inUV;


layout(location = 0) out float outDepth;

layout(push_constant) uniform PushConstants {


    layout(offset = 144) int base_color_index;
} pc;


layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {

    if (pc.base_color_index >= 0) {
        float alpha = texture(textures[nonuniformEXT(pc.base_color_index)], inUV).a;
        if (alpha < 0.1) {
            discard;
        }
    }


    outDepth = gl_FragCoord.z;
}
