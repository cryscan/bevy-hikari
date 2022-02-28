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

[[group(3), binding(0)]]
var<uniform> volume: Volume;
[[group(3), binding(1)]]
var voxel_texture: texture_storage_3d<rgba8unorm, write>;

let PI: f32 = 3.141592653589793;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn schlick(f0: f32, f90: f32, VoH: f32) -> f32 {
    return mix(f0, f90, pow(1.0 - VoH, 5.0));
}

fn burley(roughness: f32, NoV: f32, NoL: f32, LoH: f32) -> f32 {
    let f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    let light_scatter = schlick(1.0, f90, NoL);
    let view_scatter = schlick(1.0, f90, NoV);
    return light_scatter * view_scatter * (1.0 / PI);
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
    roughness: f32,
    NdotV: f32,
    normal: vec3<f32>,
    view: vec3<f32>,
    diffuse_color: vec3<f32>,
) -> vec3<f32> {
    let incident_light = light.direction_to_light.xyz;

    let half_vector = normalize(incident_light + view);
    let NoL = saturate(dot(normal, incident_light));
    let NoH = saturate(dot(normal, half_vector));
    let LoH = saturate(dot(incident_light, half_vector));

    let diffuse = diffuse_color * burley(roughness, NdotV, NoL, LoH);
    return diffuse * light.color.rgb * NoL;
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

        // if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE) != 0u) {
        output_color.a = 1.0;
        // }

        let N = normalize(in.world_normal);

        var V: vec3<f32>;
        // If the projection is not orthographic
        let is_orthographic = view.projection[3].w == 1.0;
        if (is_orthographic) {
            // Orthographic view vector
            V = normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
        } else {
            // Only valid for a perpective projection
            V = normalize(view.world_position.xyz - in.world_position.xyz);
        }

        let NdotV = max(dot(N, V), 0.0001);

        let diffuse_color = output_color.rgb * (1.0 - metallic);

        // accumulate color
        var light_accum: vec3<f32> = vec3<f32>(0.0);

        let view_z = dot(vec4<f32>(
            view.inverse_view[0].z,
            view.inverse_view[1].z,
            view.inverse_view[2].z,
            view.inverse_view[3].z,
        ), in.world_position);

        for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
            let light = lights.directional_lights[i];
            var shadow: f32 = 1.0;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                shadow = fetch_directional_shadow(i, in.world_position, in.world_normal);
            }
            let light_contrib = directional_light(light, roughness, NdotV, N, V, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }

        output_color = vec4<f32>(
            light_accum + emissive.rgb * output_color.a,
            output_color.a
        );
    }

    // tone_mapping
    output_color = vec4<f32>(reinhard_luminance(output_color.rgb), output_color.a);
    output_color = vec4<f32>(pow(output_color.rgb, vec3<f32>(1.0 / 2.2)), output_color.a);
    textureStore(voxel_texture, index, output_color);

    return output_color;
}