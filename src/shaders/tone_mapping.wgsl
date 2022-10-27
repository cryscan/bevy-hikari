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
var output_texture: texture_storage_2d<rgba8unorm, read_write>;

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

// fn tone_mapping(in: vec4<f32>) -> vec4<f32> {
//     // tone_mapping
//     return vec4<f32>(reinhard_luminance(in.rgb), in.a);
// }

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
