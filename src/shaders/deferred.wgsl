#import bevy_pbr::mesh_view_bind_group

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
var normal_map_texture: texture_2d<f32>;
[[group(1), binding(4)]]
var normal_map_sampler: sampler;

[[group(1), binding(5)]]
var position_texture: texture_storage_2d<rgba32float, write>;
[[group(1), binding(6)]]
var normal_texture: texture_storage_2d<rgba8snorm, write>;

struct FragmentInput {
    [[builtin(front_facing)]] is_front: bool;
    [[builtin(position)]] frag_coord: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
#ifdef VERTEX_TANGENTS
    [[location(3)]] world_tangent: vec4<f32>;
#endif
#ifdef VERTEX_COLORS
    [[location(4)]] color: vec4<f32>;
#endif 
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    var output_color = material.base_color;
#ifdef VERTEX_COLORS
    ououtput_color = ooutput_color * in.color;
#endif
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

    var N: vec3<f32> = normalize(in.world_normal);

#ifdef VERTEX_TANGENTS
#ifdef STANDARDMATERIAL_NORMAL_MAP
    var T: vec3<f32> = normalize(in.world_tangent.xyz - N * dot(in.world_tangent.xyz, N));
    var B: vec3<f32> = cross(N, T) * in.world_tangent.w;
#endif
#endif

    if ((material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u) {
        if (!in.is_front) {
            N = -N;
#ifdef VERTEX_TANGENTS
#ifdef STANDARDMATERIAL_NORMAL_MAP
            T = -T;
            B = -B;
#endif
#endif
        }
    }

#ifdef VERTEX_TANGENTS
#ifdef STANDARDMATERIAL_NORMAL_MAP
    let TBN = mat3x3<f32>(T, B, N);
        // Nt is the tangent-space normal.
    var Nt: vec3<f32>;
    if ((material.flags & STANDARD_MATERIAL_FLAGS_TWO_COMPONENT_NORMAL_MAP) != 0u) {
            // Only use the xy components and derive z for 2-component normal maps.
        Nt = vec3<f32>(textureSample(normal_map_texture, normal_map_sampler, in.uv).rg * 2.0 - 1.0, 0.0);
        Nt.z = sqrt(1.0 - Nt.x * Nt.x - Nt.y * Nt.y);
    } else {
        Nt = textureSample(normal_map_texture, normal_map_sampler, in.uv).rgb * 2.0 - 1.0;
    }
        // Normal maps authored for DirectX require flipping the y component
    if ((material.flags & STANDARD_MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y) != 0u) {
        Nt.y = -Nt.y;
    }
    N = normalize(TBN * Nt);
#endif
#endif

    if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE) != 0u) {
        output_color.a = 1.0;
    } else if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK) != 0u) {
        if (output_color.a >= material.alpha_cutoff) {
            output_color.a = 1.0;
        } else {
            discard;
        }
    }

    let position_coords = vec2<i32>(in.uv * vec2<f32>(textureDimensions(position_texture)));
    textureStore(position_texture, position_coords, in.world_position);

    let normal_coords = vec2<i32>(in.uv * vec2<f32>(textureDimensions(normal_map_texture)));
    textureStore(normal_texture, normal_coords, vec4<f32>(N));

    return output_color;
}