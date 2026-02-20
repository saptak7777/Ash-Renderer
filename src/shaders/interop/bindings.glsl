// Phase 9.3: IBL Descriptor Bindings
// Unified interface for Environmental Lighting (Set 0)

layout(set = 0, binding = 10) uniform samplerCube u_IrradianceMap;
layout(set = 0, binding = 11) uniform samplerCube u_PrefilterMap;
layout(set = 0, binding = 12) uniform sampler2D   u_BrdfLut;
