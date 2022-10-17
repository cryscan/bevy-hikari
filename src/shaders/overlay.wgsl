#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var direct_texture: texture_2d<f32>;
@group(2) @binding(1)
var indirect_texture: texture_2d<f32>;
@group(2) @binding(2)
var previous_render_texture: texture_2d<f32>;
@group(2) @binding(3)
var previous_render_sampler: sampler;
@group(2) @binding(4)
var temporal_texture: texture_storage_2d<rgba8unorm, write>;

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

fn tone_mapping(in: vec4<f32>) -> vec4<f32> {
    // tone_mapping
    return vec4<f32>(reinhard_luminance(in.rgb), in.a);
}

fn fetch_color(coords: vec2<i32>) -> vec3<f32> {
    let alpha = clamp(textureLoad(normal_texture, coords, 0).a, 0.0, 1.0);
    let direct = textureLoad(direct_texture, coords, 0);
    let indirect = textureLoad(indirect_texture, coords, 0);
    var color = tone_mapping(direct + indirect);
    color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    // return mix(frame.clear_color.rgb, color.rgb, alpha);
    return frame.clear_color.rgb * (1.0 - alpha) + color.rgb * alpha;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
};

@vertex
fn vertex(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 1.0);
    out.position = position;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    var uv = 0.5 * in.position.xy + 0.5;
    uv.y = 1.0 - uv.y;
    let coords = vec2<i32>(in.clip_position.xy);
    let size = textureDimensions(position_texture);

#ifdef TEMPORAL_ANTI_ALIASING
    let velocity = textureLoad(velocity_texture, coords, 0).xy;
    let previous_uv = uv - velocity;
    let previous_color = textureSample(previous_render_texture, previous_render_sampler, previous_uv);

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
    out.color = vec4<f32>(antialiased, 1.0);

    textureStore(temporal_texture, coords, vec4<f32>(antialiased, mix_rate));
#else
    let direct = textureLoad(direct_texture, coords, 0);
    let indirect = textureLoad(indirect_texture, coords, 0);
    out.color = clamp(tone_mapping(direct + indirect), vec4<f32>(0.0), vec4<f32>(1.0));
#endif

    return out;
}