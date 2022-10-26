#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var direct_render_texture: texture_2d<f32>;
@group(2) @binding(1)
var direct_render_sampler: sampler;
@group(2) @binding(2)
var emissive_render_texture: texture_2d<f32>;
@group(2) @binding(3)
var emissive_render_sampler: sampler;
@group(2) @binding(4)
var indirect_render_texture: texture_2d<f32>;
@group(2) @binding(5)
var indirect_render_sampler: sampler;

@group(3) @binding(0)
var previous_render_texture: texture_2d<f32>;
@group(3) @binding(1)
var previous_render_sampler: sampler;
@group(3) @binding(2)
var render_texture: texture_2d<f32>;
@group(3) @binding(3)
var render_texture_sampler: sampler;
@group(3) @binding(4)
var taa_output_texture: texture_storage_2d<rgba8unorm, read_write>;

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

fn fetch_color(coords: vec2<i32>) -> vec3<f32> {
    let color = textureLoad(render_texture, coords, 0);
    return mix(frame.clear_color.rgb, color.rgb, color.a);
}

@compute @workgroup_size(8, 8, 1)
fn tone_mapping(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let coords = vec2<i32>(invocation_id.xy);
    var color = vec4<f32>(0.0);

    let direct_uv = coords_to_uv(coords, textureDimensions(direct_render_texture));
    let emissive_uv = coords_to_uv(coords, textureDimensions(emissive_render_texture));
    let indirect_uv = coords_to_uv(coords, textureDimensions(indirect_render_texture));

    color += textureSampleLevel(direct_render_texture, direct_render_sampler, direct_uv, 0.0);
    color += textureSampleLevel(emissive_render_texture, emissive_render_sampler, emissive_uv, 0.0);
    color += textureSampleLevel(indirect_render_texture, indirect_render_sampler, indirect_uv, 0.0);

    color = vec4<f32>(reinhard_luminance(color.rgb), color.a);
    textureStore(output_texture, coords, color);
}

@compute @workgroup_size(8, 8, 1)
fn taa(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, textureDimensions(output_texture));

    let velocity = textureLoad(velocity_uv_texture, coords, 0).xy;
    let previous_uv = uv - velocity;
    let previous_color = textureSampleLevel(previous_render_texture, previous_render_sampler, previous_uv, 0.0);

    var antialiased = previous_color.rgb;
    var mix_rate = min(previous_color.a, 0.5);

    var colors: array<vec3<f32>, 9>;
    colors[0] = fetch_color(coords);
    antialiased = sqrt(mix(antialiased * antialiased, colors[0] * colors[0], mix_rate));

    colors[1] = fetch_color(coords + vec2<i32>(1, 0));
    colors[2] = fetch_color(coords + vec2<i32>(-1, 0));
    colors[3] = fetch_color(coords + vec2<i32>(0, 1));
    colors[4] = fetch_color(coords + vec2<i32>(0, -1));
    colors[5] = fetch_color(coords + vec2<i32>(1, 1));
    colors[6] = fetch_color(coords + vec2<i32>(-1, 1));
    colors[7] = fetch_color(coords + vec2<i32>(1, -1));
    colors[8] = fetch_color(coords + vec2<i32>(-1, -1));

    antialiased = encode_pal_yuv(antialiased);
    colors[0] = encode_pal_yuv(colors[0]);
    colors[1] = encode_pal_yuv(colors[1]);
    colors[2] = encode_pal_yuv(colors[2]);
    colors[3] = encode_pal_yuv(colors[3]);
    colors[4] = encode_pal_yuv(colors[4]);
    colors[5] = encode_pal_yuv(colors[5]);
    colors[6] = encode_pal_yuv(colors[6]);
    colors[7] = encode_pal_yuv(colors[7]);
    colors[8] = encode_pal_yuv(colors[8]);

    var min_color = min(min(min(colors[0], colors[1]), min(colors[2], colors[3])), colors[4]);
    var max_color = max(max(max(colors[0], colors[1]), max(colors[2], colors[3])), colors[4]);

    min_color = mix(min_color, min(min(min(colors[5], colors[6]), min(colors[7], colors[8])), min_color), 0.5);
    max_color = mix(max_color, max(max(max(colors[5], colors[6]), max(colors[7], colors[8])), max_color), 0.5);

    let pre_clamped = antialiased;
    antialiased = clamp(antialiased, min_color.rgb, max_color.rgb);
    mix_rate = 1.0 / (1.0 / mix_rate + 1.0);

    let delta = antialiased - pre_clamped;
    let clamp_amount = dot(delta, delta);

    mix_rate += clamp_amount * 4.0;
    mix_rate = clamp(mix_rate, 0.05, 0.5);

    antialiased = decode_pal_yuv(antialiased);
    textureStore(taa_output_texture, coords, vec4<f32>(antialiased, mix_rate));
    textureStore(output_texture, coords, vec4<f32>(antialiased, 1.0));
}
