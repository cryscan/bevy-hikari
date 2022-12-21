#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings

#import bevy_core_pipeline::fullscreen_vertex_shader

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var radiance_texture: texture_2d<f32>;
@group(3) @binding(1)
var albedo_texture: texture_2d<f32>;

fn jittered_deferred_uv(uv: vec2<f32>, deferred_texel_size: vec2<f32>) -> vec2<f32> {
    return uv + select(0.5, -0.5, (frame.number & 1u) == 0u) * deferred_texel_size;
}

@fragment
fn modulation(uv: vec2<f32>) -> @location(0) vec4<f32> {
    let radiance = textureSample(radiance_texture, nearest_sampler, uv);

    let deferred_texel_size = 1.0 / view.viewport.zw;
    let deferred_uv = jittered_deferred_uv(uv, deferred_texel_size);
    let albedo = textureSample(albedo_texture, nearest_sampler, deferred_uv);

    return vec4<f32>(radiance.rgb * albedo.rgb, albedo.a);
}