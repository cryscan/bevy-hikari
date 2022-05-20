#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct
#import bevy_hikari::volume_struct
#import bevy_hikari::standard_material

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

[[group(3), binding(0)]]
var<uniform> volume: Volume;
[[group(3), binding(1)]]
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

// fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
//     let l_in = luminance(c_in);
//     return c_in * (l_out / l_in);
// }

fn reinhard_luminance(color: vec3<f32>) -> vec4<f32> {
    let lum = luminance(color);
    let factor = 1.0 / (1.0 + lum);
    // return change_luminance(color, l_new);
    return vec4<f32>(color * factor, 1.0 / factor);
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

fn spatial_index(position: vec3<f32>) -> vec3<i32> {
    let dims = vec3<u32>(VOXEL_SIZE - 1u);
    let coords = (position - volume.min) / (volume.max - volume.min);
    return vec3<i32>(0.5 + vec3<f32>(dims) * coords);
}

fn linear_index(index: vec3<i32>) -> i32 {
    var spatial = vec3<u32>(index);
    var morton = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let coords = spatial & vec3<u32>(1u);
        let offset = 3u * i;

        morton = morton | (coords.x << offset);
        morton = morton | (coords.y << (offset + 1u));
        morton = morton | (coords.z << (offset + 2u));

        spatial = spatial >> vec3<u32>(1u);
    }

    return i32(morton);
}

struct FragmentInput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    if ((material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) {
        discard;
    }
    if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND) != 0u) {
        discard;
    }

    var output_color = material.base_color;
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

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

    let clip_normal = abs(face_normal(in.clip_position));
    if (clip_normal.z < max(clip_normal.x, clip_normal.y)) {
        discard;
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

    let normal = normalize(in.world_normal);
    let diffuse_color = output_color.rgb * (1.0 - metallic);
    var light_accum: vec3<f32> = vec3<f32>(0.0);

    for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let light = lights.directional_lights[i];
        var shadow: f32 = 1.0;
        if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = fetch_directional_shadow(i, in.world_position, in.world_normal);
        }
        let light_contrib = directional_light(light, normal, diffuse_color);
        light_accum = light_accum + light_contrib * shadow;
    }

    output_color = vec4<f32>(
        (light_accum + emissive.rgb) * output_color.a,
        output_color.a
    );

    // Tone mapping, but keep HDR info.
    let color = reinhard_luminance(output_color.rgb);
    let packed = pack4x8unorm(vec4<f32>(color.rgb, color.a / 255.0));

    let index = spatial_index(in.world_position.xyz);
    let voxel = &voxel_buffer.data[linear_index(index)];
    atomicMax(voxel, packed);

    return output_color;
}