#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings

#import bevy_core_pipeline::fullscreen_vertex_shader

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@fragment
fn nearest_velocity(uv: vec2<f32>) -> @location(0) vec2<f32> {
    let texel_size = 1.0 / view.viewport.zw;

    var depths: vec4<f32>;
    depths[0] = textureSample(position_texture, nearest_sampler, uv + texel_size * vec2<f32>(-1.0, -1.0)).w;
    depths[1] = textureSample(position_texture, nearest_sampler, uv + texel_size * vec2<f32>(1.0, -1.0)).w;
    depths[2] = textureSample(position_texture, nearest_sampler, uv + texel_size * vec2<f32>(-1.0, 1.0)).w;
    depths[3] = textureSample(position_texture, nearest_sampler, uv + texel_size * vec2<f32>(1.0, 1.0)).w;
    let max_depth = max(max(depths[0], depths[1]), max(depths[2], depths[3]));

    let depth = textureSample(position_texture, nearest_sampler, uv).w;
    var offset = vec2<f32>(0.0);
    if depth < max_depth {
        let x = dot(vec4<f32>(texel_size.x), select(vec4<f32>(0.0), vec4<f32>(1.0, -1.0, 1.0, -1.0), depths == vec4<f32>(max_depth)));
        let y = dot(vec4<f32>(texel_size.y), select(vec4<f32>(0.0), vec4<f32>(1.0, 1.0, -1.0, -1.0), depths == vec4<f32>(max_depth)));
        offset = vec2<f32>(x, y);
    }

    return textureSample(velocity_uv_texture, nearest_sampler, uv + offset).xy;
}