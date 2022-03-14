#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

struct StandardMaterial {
    base_color: vec4<f32>;
    emissive: vec4<f32>;
    perceptual_roughness: f32;
    metallic: f32;
    reflectance: f32;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
    alpha_cutoff: f32;
};

let STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT: u32 = 1u;
let STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT: u32 = 2u;
let STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 4u;
let STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT: u32 = 8u;
let STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT: u32 = 16u;
let STANDARD_MATERIAL_FLAGS_UNLIT_BIT: u32 = 32u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE: u32 = 64u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK: u32 = 128u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND: u32 = 256u;

[[group(1), binding(0)]]
var<uniform> material: StandardMaterial;
[[group(1), binding(1)]]
var base_color_texture: texture_2d<f32>;
[[group(1), binding(2)]]
var base_color_sampler: sampler;
[[group(1), binding(3)]]
var emissive_texture: texture_2d<f32>;
[[group(1), binding(4)]]
var emissive_sampler: sampler;
[[group(1), binding(5)]]
var metallic_roughness_texture: texture_2d<f32>;
[[group(1), binding(6)]]
var metallic_roughness_sampler: sampler;
[[group(1), binding(7)]]
var occlusion_texture: texture_2d<f32>;
[[group(1), binding(8)]]
var occlusion_sampler: sampler;
[[group(1), binding(9)]]
var normal_map_texture: texture_2d<f32>;
[[group(1), binding(10)]]
var normal_map_sampler: sampler;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
};

struct Voxel {
    top: atomic<u32>;
    bot: atomic<u32>;
};

struct VoxelBuffer {
    data: array<Voxel>;
};

[[group(3), binding(0)]]
var<uniform> volume: Volume;
[[group(3), binding(1)]]
var voxel_texture: texture_3d<f32>;
[[group(3), binding(2)]]
var<storage, read_write> voxel_buffer: VoxelBuffer;

let PI: f32 = 3.141592653589793;

fn face_normal(clip_position: vec4<f32>) -> vec3<f32> {
    let position = clip_position.xyz / clip_position.w;
    return cross(dpdx(position), dpdy(position));
}

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn compute_roughness(perceptual_roughness: f32) -> f32 {
    let clamped = clamp(perceptual_roughness, 0.089, 1.0);
    return clamped * clamped;
}

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    let l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let l_old = luminance(color);
    let l_new = l_old / (1.0 + l_old);
    return change_luminance(color, l_new);
}

fn directional_light(
    light: DirectionalLight,
    normal: vec3<f32>,
    diffuse_color: vec3<f32>,
) -> vec3<f32> {
    let incident_light = light.direction_to_light.xyz;
    let NoL = saturate(dot(normal, incident_light));
    return diffuse_color * light.color.rgb * NoL;
}

fn fetch_directional_shadow(light_id: u32, frag_position: vec4<f32>, surface_normal: vec3<f32>) -> f32 {
    let light = lights.directional_lights[light_id];

    // The normal bias is scaled to the texel size.
    let normal_offset = light.shadow_normal_bias * surface_normal.xyz;
    let depth_offset = light.shadow_depth_bias * light.direction_to_light.xyz;
    let offset_position = vec4<f32>(frag_position.xyz + normal_offset + depth_offset, frag_position.w);

    let offset_position_clip = light.view_projection * offset_position;
    if (offset_position_clip.w <= 0.0) {
        return 1.0;
    }
    let offset_position_ndc = offset_position_clip.xyz / offset_position_clip.w;
    // No shadow outside the orthographic projection volume
    if (any(offset_position_ndc < vec3<f32>(-1.0, -1.0, 0.0)) || any(offset_position_ndc > vec3<f32>(1.0))) {
        return 1.0;
    }

    // compute texture coordinates for shadow lookup, compensating for the Y-flip difference
    // between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    let light_local = offset_position_ndc.xy * flip_correction + vec2<f32>(0.5, 0.5);

    let depth = offset_position_ndc.z;
    // do the lookup, using HW PCF and comparison
    // NOTE: Due to non-uniform control flow above, we must use the level variant of the texture
    // sampler to avoid use of implicit derivatives causing possible undefined behavior.
#ifdef NO_ARRAY_TEXTURES_SUPPORT
    return textureSampleCompareLevel(directional_shadow_textures, directional_shadow_textures_sampler, light_local, depth);
#else
    return textureSampleCompareLevel(directional_shadow_textures, directional_shadow_textures_sampler, light_local, i32(light_id), depth);
#endif
}

fn linear_index(index: vec3<i32>) -> i32 {
    let dims = textureDimensions(voxel_texture);
    return index.x + index.y * dims.x + index.z * dims.x * dims.y;
}

struct FragmentInput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    let coords = (in.world_position.xyz - volume.min) / (volume.max - volume.min);
    let index = vec3<i32>(0.5 + vec3<f32>(textureDimensions(voxel_texture)) * coords);

    var output_color = material.base_color;
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

    if ((material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        var emissive = material.emissive;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT) != 0u) {
            emissive = emissive * textureSample(emissive_texture, emissive_sampler, in.uv);
            emissive.a = 1.0;
        }

        var metallic = material.metallic;
        var perceptual_roughness = material.perceptual_roughness;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u) {
            let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
            metallic = metallic * metallic_roughness.b;
            perceptual_roughness = perceptual_roughness * metallic_roughness.g;
        }
        let roughness = compute_roughness(perceptual_roughness);

        var occlusion: f32 = 1.0;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT) != 0u) {
            occlusion = textureSample(occlusion_texture, occlusion_sampler, in.uv).r;
        }

        if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE) != 0u) {
            output_color.a = 1.0;
        } else if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK) != 0u) {
            if (output_color.a >= material.alpha_cutoff) {
                output_color.a = 1.0;
            } else {
                discard;
            }
        }

        let N = normalize(in.world_normal);

        let diffuse_color = output_color.rgb * (1.0 - metallic);

        // accumulate color
        var light_accum: vec3<f32> = vec3<f32>(0.0);

        for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
            let light = lights.directional_lights[i];
            var shadow: f32 = 1.0;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                shadow = fetch_directional_shadow(i, in.world_position, in.world_normal);
            }
            let light_contrib = directional_light(light, N, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }

        output_color = vec4<f32>(
            (light_accum + emissive.rgb) * output_color.a,
            output_color.a
        );
    }

    let clip_normal = abs(face_normal(in.clip_position));
    if (clip_normal.z < max(clip_normal.x, clip_normal.y)) {
        discard;
    }

    // tone_mapping
    output_color = vec4<f32>(reinhard_luminance(output_color.rgb), output_color.a);
    
    let voxel = &voxel_buffer.data[linear_index(index)];
    let converted = vec4<u32>((output_color + 0.5) * 255.);
    atomicAdd(&(*voxel).top, (converted.r << 16u) + converted.g);
    atomicAdd(&(*voxel).bot, (converted.b << 16u) + converted.a);

    return output_color;
}