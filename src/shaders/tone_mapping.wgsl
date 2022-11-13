#import bevy_core_pipeline::tonemapping
#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var direct_render_texture: texture_2d<f32>;
@group(3) @binding(1)
var emissive_render_texture: texture_2d<f32>;
@group(3) @binding(2)
var indirect_render_texture: texture_2d<f32>;

@group(4) @binding(0)
var output_texture: texture_storage_2d<rgba16float, read_write>;

@compute @workgroup_size(8, 8, 1)
fn tone_mapping(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let coords = vec2<i32>(invocation_id.xy);

    var color = textureLoad(direct_render_texture, coords, 0);
    color += textureLoad(emissive_render_texture, coords, 0);
    color += textureLoad(indirect_render_texture, coords, 0);

    color = vec4<f32>(reinhard_luminance(color.rgb), color.a);
    color = select(frame.clear_color, color, color.a > 0.0);
    textureStore(output_texture, coords, color);
}
