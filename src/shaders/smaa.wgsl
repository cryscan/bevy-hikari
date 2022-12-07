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
@group(3) @binding(2)
var albedo_texture: texture_2d<f32>;

@group(4) @binding(0)
var output_texture: texture_storage_2d<rgba16float, read_write>;

let TAU: f32 = 6.283185307;
let INV_TAU: f32 = 0.159154943;

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

fn sample_previous_texture(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(previous_render_texture, linear_sampler, uv, 0.0).rgb;
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_instance(uv: vec2<f32>, offset: vec2<f32>) -> u32 {
    let size = vec2<f32>(textureDimensions(instance_material_texture));
    let coords = vec2<i32>((uv + offset) * size);
    return textureLoad(previous_instance_material_texture, coords, 0).x;
}

fn nearest_velocity(uv: vec2<f32>) -> vec2<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(render_texture));
    var offset = vec2<f32>(0.0);
    var depth = textureSampleLevel(position_texture, nearest_sampler, uv, 0.0).w;

    var offset_next = vec2<f32>(texel_size.x, texel_size.y);
    offset = select(offset, offset_next, textureSampleLevel(position_texture, nearest_sampler, offset_next, 0.0).w > depth);
    offset_next = vec2<f32>(-texel_size.x, texel_size.y);
    offset = select(offset, offset_next, textureSampleLevel(position_texture, nearest_sampler, offset_next, 0.0).w > depth);
    offset_next = vec2<f32>(texel_size.x, -texel_size.y);
    offset = select(offset, offset_next, textureSampleLevel(position_texture, nearest_sampler, offset_next, 0.0).w > depth);
    offset_next = vec2<f32>(-texel_size.x, -texel_size.y);
    offset = select(offset, offset_next, textureSampleLevel(position_texture, nearest_sampler, offset_next, 0.0).w > depth);

    return textureSampleLevel(velocity_uv_texture, nearest_sampler, uv + offset, 0.0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn smaa_tu4x(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let input_size = textureDimensions(render_texture);
    let output_size = textureDimensions(output_texture);
    let deferred_size = textureDimensions(position_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, input_size);

    // In this implementation, a thread computes 4 output pixels in a quad.
    // One of the pixel (c) on the diagonal can be fetched from render_texture,
    // the other (p) is reprojected from previous_render_texture.
    // The two left (p', c') are extrapolated using differential blend method.

    // Odd frame:
    //     -1  0  1  2
    //    +--+--+--+--+
    // -1 |  |  |cn|  |
    //  0 |  |p |c'|pe|
    //  1 |cw|p'|c |  |
    //  2 |  |ps|  |  |
    //    +--+--+--+--+

    // Even frame:
    //     -1  0  1  2
    //    +--+--+--+--+
    // -1 |  |  |pn|  |
    //  0 |  |c |p'|ce|
    //  1 |pw|c'|p |  |
    //  2 |  |cs|  |  |
    //    +--+--+--+--+

    // Reproject to find the equivalent sample from the past, using 5-tap Catmull-Rom filtering
    // from https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // and https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
    let texel_size = 1.0 / vec2<f32>(output_size);
    let tile_size = 1.0 / vec2<f32>(input_size);

    // Fetch the current sample
    let current_output_coords = 2 * coords + current_smaa_jitter(frame.number);
    let current_output_uv = coords_to_uv(current_output_coords, output_size);
    let current_albedo = textureSampleLevel(albedo_texture, nearest_sampler, current_output_uv, 0.0);
    let current_color = textureSampleLevel(render_texture, nearest_sampler, uv, 0.0).rgb;

    // Fetch the previous sample
    let previous_output_coords = 2 * coords + previous_smaa_jitter(frame.number);
    let previous_output_uv = coords_to_uv(previous_output_coords, output_size);
    let previous_albedo = textureSampleLevel(albedo_texture, nearest_sampler, previous_output_uv, 0.0);
    let velocity = nearest_velocity(previous_output_uv);
    let previous_reprojected_uv = previous_output_uv - velocity;

    // let sample_position = previous_reprojected_uv * vec2<f32>(input_size);
    // let texel_position_1 = floor(sample_position - 0.5) + 0.5;
    // let f = sample_position - texel_position_1;
    // let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    // let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    // let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    // let w3 = f * f * (-0.5 + 0.5 * f);
    // let w12 = w1 + w2;
    // let offset12 = w2 / (w1 + w2);
    // let texel_position_0 = (texel_position_1 - 1.0) * tile_size;
    // let texel_position_3 = (texel_position_1 + 2.0) * tile_size;
    // let texel_position_12 = (texel_position_1 + offset12) * tile_size;
    // var previous_color = vec3<f32>(0.0);
    // previous_color += sample_previous_texture(vec2<f32>(texel_position_12.x, texel_position_0.y)) * w12.x * w0.y;
    // previous_color += sample_previous_texture(vec2<f32>(texel_position_0.x, texel_position_12.y)) * w0.x * w12.y;
    // previous_color += sample_previous_texture(vec2<f32>(texel_position_12.x, texel_position_12.y)) * w12.x * w12.y;
    // previous_color += sample_previous_texture(vec2<f32>(texel_position_3.x, texel_position_12.y)) * w3.x * w12.y;
    // previous_color += sample_previous_texture(vec2<f32>(texel_position_12.x, texel_position_3.y)) * w12.x * w3.y;
    var previous_color = textureSampleLevel(previous_render_texture, nearest_sampler, previous_reprojected_uv, 0.0).rgb;

    let current_depth = textureSampleLevel(position_texture, nearest_sampler, previous_output_uv, 0.0).w;
    let previous_depths = textureGather(3, previous_position_texture, linear_sampler, previous_reprojected_uv);
    let depth_ratio = vec4<f32>(current_depth) / max(previous_depths, vec4<f32>(0.0001));
    let depth_miss = current_depth == 0.0 || any(depth_ratio < vec4<f32>(0.95)) || any(depth_ratio > vec4<f32>(1.05));

    let current_instance = textureLoad(instance_material_texture, previous_output_coords, 0).x;
    var instance_miss = current_instance != sample_instance(previous_reprojected_uv, vec2<f32>(0.0));
    instance_miss = instance_miss || current_instance != sample_instance(previous_reprojected_uv, vec2<f32>(-tile_size.x, -tile_size.y));
    instance_miss = instance_miss || current_instance != sample_instance(previous_reprojected_uv, vec2<f32>(tile_size.x, -tile_size.y));
    instance_miss = instance_miss || current_instance != sample_instance(previous_reprojected_uv, vec2<f32>(-tile_size.x, tile_size.y));
    instance_miss = instance_miss || current_instance != sample_instance(previous_reprojected_uv, vec2<f32>(tile_size.x, tile_size.y));

    let previous_velocity = textureSampleLevel(previous_velocity_uv_texture, nearest_sampler, previous_reprojected_uv, 0.0).xy;
    let velocity_miss = distance(velocity, previous_velocity) > 0.0001;

    if (depth_miss || instance_miss) && velocity_miss {
        // Constrain past sample with 2x2 YCoCg variance clipping to handle disocclusion
        let cr = textureGather(0, render_texture, linear_sampler, previous_output_uv);
        let cg = textureGather(1, render_texture, linear_sampler, previous_output_uv);
        let cb = textureGather(2, render_texture, linear_sampler, previous_output_uv);
        let s1 = RGB_to_YCoCg(vec3<f32>(cr.x, cg.x, cb.x));
        let s2 = RGB_to_YCoCg(vec3<f32>(cr.y, cg.y, cb.y));
        let s3 = RGB_to_YCoCg(vec3<f32>(cr.z, cg.z, cb.z));
        let s4 = RGB_to_YCoCg(vec3<f32>(cr.w, cg.w, cb.w));
        let s_mm = RGB_to_YCoCg(current_color);
        let moment_1 = s1 + s2 + s3 + s4;
        let moment_2 = s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4;
        let mean = moment_1 / 4.0;
        let variance = sqrt((moment_2 / 4.0) - (mean * mean));
        previous_color = RGB_to_YCoCg(previous_color);
        previous_color = clip_towards_aabb_center(previous_color, s_mm, mean - variance, mean + variance);
        previous_color = YCoCg_to_RGB(previous_color);
    }

    // Get the subpixel velocity,
    // And blend the previous sample with the approximated current to get better edges
    let subpixel_velocity = fract(velocity / tile_size);
    var blend_factor = max(subpixel_velocity.x, subpixel_velocity.y);
    blend_factor = clamp(-cos(blend_factor * TAU), 0.0, 1.0);

    let remix_color = textureSampleLevel(render_texture, linear_sampler, previous_output_uv, 0.0).rgb;
    previous_color = mix(previous_color, remix_color, blend_factor);

    textureStore(output_texture, current_output_coords, vec4<f32>(current_color, 1.0));
    textureStore(output_texture, previous_output_coords, vec4<f32>(previous_color, 1.0));
}

fn differential_blend_factor(
    t: vec4<f32>,
    b: vec4<f32>,
    n: vec4<f32>,
    e: vec4<f32>,
    s: vec4<f32>,
    w: vec4<f32>
) -> vec3<f32> {
    let dh = vec2<f32>(
        luminance(abs(w.rgb - b.rgb)),
        luminance(abs(t.rgb - e.rgb))
    );
    let dv = vec2<f32>(
        luminance(abs(t.rgb - s.rgb)),
        luminance(abs(n.rgb - b.rgb))
    );

    let factor_xy = vec2<f32>(
        max(dv.x, 0.001) * max(dv.y, 0.001),
        max(dh.x, 0.001) * max(dh.y, 0.001),
    );
    let factor_z = 1.0 / (factor_xy.x + factor_xy.y);
    return vec3<f32>(factor_xy, factor_z);
}

fn differential_blend(
    t: vec4<f32>,
    b: vec4<f32>,
    l: vec4<f32>,
    r: vec4<f32>,
    factor: vec3<f32>,
) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    color += (l + r) * factor.x;
    color += (t + b) * factor.y;
    return (0.5 * factor.z) * color;
}

@compute @workgroup_size(8, 8, 1)
fn smaa_tu4x_extrapolate(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let input_size = textureDimensions(render_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, input_size);

    let frame_index = frame.number % 2u;

    // Fetch neighbour colors
    //     -1  0  1  2
    //    +--+--+--+--+
    // -1 |  |  |n |  |
    //  0 |  |t |y |e |
    //  1 |w |x |b |  |
    //  2 |  |s |  |  |
    //    +--+--+--+--+

    let t_color = textureLoad(output_texture, 2 * coords);
    let b_color = textureLoad(output_texture, 2 * coords + vec2<i32>(1, 1));
    let n_color = textureLoad(output_texture, 2 * coords + vec2<i32>(1, -1));
    let e_color = textureLoad(output_texture, 2 * coords + vec2<i32>(2, 0));
    let s_color = textureLoad(output_texture, 2 * coords + vec2<i32>(0, 2));
    let w_color = textureLoad(output_texture, 2 * coords + vec2<i32>(-1, 1));

    let factor = differential_blend_factor(t_color, b_color, n_color, e_color, s_color, w_color);
    let x_color = differential_blend(t_color, s_color, w_color, b_color, factor);
    let y_color = differential_blend(n_color, b_color, t_color, e_color, factor);

    // textureStore(output_texture, 2 * coords, t_color);
    // textureStore(output_texture, 2 * coords + vec2<i32>(1, 1), b_color);
    textureStore(output_texture, 2 * coords + vec2<i32>(0, 1), x_color);
    textureStore(output_texture, 2 * coords + vec2<i32>(1, 0), y_color);
}
