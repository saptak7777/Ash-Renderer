// VSM Common Functions - Shared between compute and fragment shaders
// Clipmap selection and virtual page calculation

// Select appropriate clipmap level based on world position
// Requires ClipmapData uniform to be bound
int selectClipmapLevel(vec3 worldPos, uint active_levels, vec4 level_centers[16]) {
    // If no clipmaps active, use layer 0
    if (active_levels == 0) {
        return 0;
    }
    
    // Start from finest level (0) and find first level that contains position
    for (int i = 0; i < int(active_levels); i++) {
        vec4 levelData = level_centers[i];
        vec2 center = levelData.xy;
        float extent = levelData.z;
        
        // Check if worldPos is within this level's bounds
        vec2 delta = abs(worldPos.xz - center);
        if (delta.x <= extent * 0.5 && delta.y <= extent * 0.5) {
            return i;
        }
    }
    
    // If outside all levels, use coarsest level
    return int(active_levels) - 1;
}
