# Shader Compilation Script for Ash Renderer
# Compiles all shaders to SPIR-V with optimizations

$ErrorActionPreference = "Stop"

Write-Host "=== Ash Renderer Shader Compilation ===" -ForegroundColor Cyan
Write-Host ""

# Check for glslc
try {
    $glslcVersion = glslc --version 2>&1 | Select-Object -First 1
    Write-Host "Using: $glslcVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: glslc not found. Please install Vulkan SDK." -ForegroundColor Red
    exit 1
}

Write-Host ""
Set-Location $PSScriptRoot\shaders

# Compile vertex shaders
Write-Host "Compiling vertex shaders..." -ForegroundColor Yellow
glslc vert.vert -o vert.spv
glslc postprocess.vert -o postprocess.vert.spv
glslc shadow.vert -o shadow.vert.spv
glslc overlay.vert -o overlay.vert.spv

# Compile fragment shaders
Write-Host "Compiling fragment shaders..." -ForegroundColor Yellow
Write-Host "  - frag.frag (WITH Forward+ lighting)" -ForegroundColor Cyan
glslc -fshader-stage=fragment -DENABLE_POINT_LIGHTS frag.frag -o frag.spv

glslc tonemapping.frag -o tonemapping.frag.spv
glslc shadow.frag -o shadow.frag.spv
glslc overlay.frag -o overlay.frag.spv

# Compile compute shaders
Write-Host "Compiling compute shaders..." -ForegroundColor Yellow
glslc light_culling.comp -o light_culling.spv
glslc taa_resolve.comp -o taa_resolve.spv
glslc occlusion_cull.comp -o occlusion_cull.spv
glslc hiz_generate.comp -o hiz_generate.spv

# Compile bloom shaders
Write-Host "Compiling bloom shaders..." -ForegroundColor Yellow
glslc bloom_prefilter.frag -o bloom_prefilter.frag.spv
glslc bloom_downsample.frag -o bloom_downsample.frag.spv
glslc bloom_upsample.frag -o bloom_upsample.frag.spv

# Compile PBR shaders
Write-Host "Compiling PBR shaders..." -ForegroundColor Yellow
glslc ../src/shaders/ibl/brdf_lut.frag -o brdf_lut.spv
glslc ../src/shaders/ibl/fullscreen.vert -o ibl_fullscreen.vert.spv
glslc ../src/shaders/ibl/skybox.vert -o ibl_skybox.vert.spv
glslc ../src/shaders/ibl/equirect_to_cube.frag -o equirect_to_cube.frag.spv
glslc ../src/shaders/ibl/irradiance.frag -o irradiance.frag.spv
glslc ../src/shaders/ibl/prefilter.frag -o prefilter.frag.spv

Write-Host ""
Write-Host "=== Compilation Complete ===" -ForegroundColor Green
Write-Host "  Forward+ lighting: ENABLED" -ForegroundColor Cyan
Write-Host ""
