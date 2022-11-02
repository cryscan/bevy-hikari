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

// The following 3 functions are from Playdead
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3<f32>(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
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

fn sample_previous_render_texture(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(previous_render_texture, linear_sampler, uv, 0.0).rgb;
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
    var previous_color = vec3<f32>(0.0);
    previous_color += sample_previous_render_texture(vec2<f32>(texel_position_12.x, texel_position_0.y)) * w12.x * w0.y;
    previous_color += sample_previous_render_texture(vec2<f32>(texel_position_0.x, texel_position_12.y)) * w0.x * w12.y;
    previous_color += sample_previous_render_texture(vec2<f32>(texel_position_12.x, texel_position_12.y)) * w12.x * w12.y;
    previous_color += sample_previous_render_texture(vec2<f32>(texel_position_3.x, texel_position_12.y)) * w3.x * w12.y;
    previous_color += sample_previous_render_texture(vec2<f32>(texel_position_12.x, texel_position_3.y)) * w12.x * w3.y;

    // Constrain past sample with 3x3 YCoCg variance clipping to handle disocclusion
    let s_tl = sample_render_texture(uv + vec2<f32>(-texel_size.x, texel_size.y));
    let s_tm = sample_render_texture(uv + vec2<f32>(0.0, texel_size.y));
    let s_tr = sample_render_texture(uv + texel_size);
    let s_ml = sample_render_texture(uv - vec2<f32>(texel_size.x, 0.0));
    let s_mm = RGB_to_YCoCg(current_color);
    let s_mr = sample_render_texture(uv + vec2<f32>(texel_size.x, 0.0));
    let s_bl = sample_render_texture(uv - texel_size);
    let s_bm = sample_render_texture(uv - vec2<f32>(0.0, texel_size.y));
    let s_br = sample_render_texture(uv + vec2<f32>(texel_size.x, -texel_size.y));
    let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    let moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) + (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) + (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
    let mean = moment_1 / 9.0;
    let variance = sqrt((moment_2 / 9.0) - (mean * mean));
    let previous_color = RGB_to_YCoCg(previous_color);
    let previous_color = clip_towards_aabb_center(previous_color, s_mm, mean - variance, mean + variance);
    let previous_color = YCoCg_to_RGB(previous_color);

    // Blend current and past sample
    let output = mix(previous_color, current_color, 0.1);

    // return vec4<f32>(output, original_color.a);
    // textureStore(accumulation_texture, coords, vec4<f32>(output, original_color.a));
    textureStore(output_texture, coords, vec4<f32>(output, original_color.a));
}
