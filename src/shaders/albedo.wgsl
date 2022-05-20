#import bevy_pbr::mesh_view_bind_group
#import bevy_hikari::standard_material

struct FragmentInput {
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
    output_color = output_color * in.color;
#endif
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

    return output_color;
}