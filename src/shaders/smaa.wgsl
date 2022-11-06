#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var previous_render_texture: texture_2d<f32>;
@group(3) @binding(1)
var render_texture: texture_2d<f32>;

@group(4) @binding(0)
var output_texture: texture_storage_2d<rgba16float, read_write>;

@compute @workgroup_size(8, 8, 1)
fn smaa_tu4x(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let input_size = textureDimensions(render_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, input_size);

    // In this implementation, a thread computes 4 output pixels in a quad.
    // One of the pixel (c) on the diagonal can be fetched from render_texture,
    // the other (p) is reprojected from previous_render_texture.
    // The two left (p', c') are extrapolated using differential blend method.

    // +--+--+--+--+
    // |  |  |cn|  |
    // |  |p |c'|pe|
    // |cw|p'|c |  |
    // |  |ps|  |  |
    // +--+--+--+--+

    let output_size = textureDimensions(output_texture);
}