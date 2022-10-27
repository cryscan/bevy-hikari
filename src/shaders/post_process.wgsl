#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var direct_render_texture: texture_2d<f32>;
@group(2) @binding(1)
var emissive_render_texture: texture_2d<f32>;
@group(2) @binding(2)
var indirect_render_texture: texture_2d<f32>;
@group(2) @binding(3)
var nearest_sampler: sampler;
@group(2) @binding(4)
var linear_sampler: sampler;

@group(3) @binding(0)
var previous_render_texture: texture_2d<f32>;
@group(3) @binding(1)
var render_texture: texture_2d<f32>;
@group(3) @binding(2)
var accumulation_texture: texture_storage_2d<rgba8unorm, read_write>;

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

    color += textureSampleLevel(direct_render_texture, nearest_sampler, direct_uv, 0.0);
    color += textureSampleLevel(emissive_render_texture, nearest_sampler, emissive_uv, 0.0);
    color += textureSampleLevel(indirect_render_texture, nearest_sampler, indirect_uv, 0.0);

    color = vec4<f32>(reinhard_luminance(color.rgb), color.a);
    textureStore(output_texture, coords, color);
}

@compute @workgroup_size(8, 8, 1)
fn taa(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, textureDimensions(output_texture));

    let velocity = textureLoad(velocity_uv_texture, coords, 0).xy;
    let previous_uv = uv - velocity;
    let previous_color = textureSampleLevel(previous_render_texture, linear_sampler, previous_uv, 0.0);

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
    textureStore(accumulation_texture, coords, vec4<f32>(antialiased, mix_rate));
    textureStore(output_texture, coords, vec4<f32>(antialiased, 1.0));
}

// The following 3 functions are from Playdead
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

fn clip_towards_aabb_center(previous_color: vec3<f32>, current_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min);
    let v_clip = previous_color - p_clip;
    let v_unit = v_clip.xyz / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    return select(previous_color, p_clip + v_clip / ma_unit, ma_unit > 1.0);
}

fn sample_render_texture(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(render_texture, nearest_sampler, uv, 0.0).rgb;
    return RGB_to_YCoCg(c);
}

@compute @workgroup_size(8, 8, 1)
fn jasmine_taa(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = vec2<f32>(textureDimensions(output_texture));
    let texel_size = 1.0 / size;

    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, textureDimensions(output_texture));

    // Fetch the current sample
    let original_color = textureSampleLevel(render_texture, nearest_sampler, uv, 0.0);
    let current_color = original_color.rgb;

    // Reproject to find the equivalent sample from the past, using 5-tap Catmull-Rom filtering
    // from https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // and https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
    let current_velocity = textureSampleLevel(velocity_uv_texture, nearest_sampler, uv, 0.0).rg;
    let sample_position = (uv - current_velocity) * size;
    let texel_position_1 = floor(sample_position - 0.5) + 0.5;
    let f = sample_position - texel_position_1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / (w1 + w2);
    let texel_position_0 = (texel_position_1 - 1.0) * texel_size;
    let texel_position_3 = (texel_position_1 + 2.0) * texel_size;
    let texel_position_12 = (texel_position_1 + offset12) * texel_size;
    var previous_color = vec3(0.0);
    previous_color += textureSampleLevel(previous_render_texture, linear_sampler, vec2(texel_position_12.x, texel_position_0.y), 0.0).rgb * w12.x * w0.y;
    previous_color += textureSampleLevel(previous_render_texture, linear_sampler, vec2(texel_position_0.x, texel_position_12.y), 0.0).rgb * w0.x * w12.y;
    previous_color += textureSampleLevel(previous_render_texture, linear_sampler, vec2(texel_position_12.x, texel_position_12.y), 0.0).rgb * w12.x * w12.y;
    previous_color += textureSampleLevel(previous_render_texture, linear_sampler, vec2(texel_position_3.x, texel_position_12.y), 0.0).rgb * w3.x * w12.y;
    previous_color += textureSampleLevel(previous_render_texture, linear_sampler, vec2(texel_position_12.x, texel_position_3.y), 0.0).rgb * w12.x * w3.y;

    // Constrain past sample with 3x3 YCoCg variance clipping to handle disocclusion
    let s_tl = sample_render_texture(uv + vec2(-texel_size.x, texel_size.y));
    let s_tm = sample_render_texture(uv + vec2(0.0, texel_size.y));
    let s_tr = sample_render_texture(uv + texel_size);
    let s_ml = sample_render_texture(uv - vec2(texel_size.x, 0.0));
    let s_mm = RGB_to_YCoCg(current_color);
    let s_mr = sample_render_texture(uv + vec2(texel_size.x, 0.0));
    let s_bl = sample_render_texture(uv - texel_size);
    let s_bm = sample_render_texture(uv - vec2(0.0, texel_size.y));
    let s_br = sample_render_texture(uv + vec2(texel_size.x, -texel_size.y));
    let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    let moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) + (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) + (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
    let mean = moment_1 / 9.0;
    let variance = sqrt((moment_2 / 9.0) - (mean * mean));
    let previous_color = RGB_to_YCoCg(previous_color);
    let previous_color = clip_towards_aabb_center(previous_color, s_mm, mean - variance, mean + variance);
    let previous_color = YCoCg_to_RGB(previous_color);

    // Blend current and past sample
    let output = (current_color * 0.1) + (previous_color * 0.9);

    // return vec4<f32>(output, original_color.a);
    textureStore(accumulation_texture, coords, vec4<f32>(output, original_color.a));
    textureStore(output_texture, coords, vec4<f32>(output, original_color.a));
}
