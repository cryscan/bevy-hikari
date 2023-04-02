#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var internal_texture_0: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(1)
var internal_texture_1: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(2)
var internal_texture_2: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(3)
var internal_texture_3: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(4)
var internal_variance: texture_storage_2d<r32float, read_write>;

@group(4) @binding(0)
var albedo_texture: texture_2d<f32>;
@group(4) @binding(1)
var variance_texture: texture_2d<f32>;
@group(4) @binding(2)
var render_texture: texture_2d<f32>;
@group(4) @binding(3)
var output_texture: texture_storage_2d<rgba16float, read_write>;

const TAU: f32 = 6.283185307;
const GOLDEN_RATIO: f32 = 1.618033989;

const F32_EPSILON: f32 = 1.1920929E-7;
const F32_MAX: f32 = 3.402823466E+38;
const U32_MAX: u32 = 0xFFFFFFFFu;

fn jittered_deferred_uv(uv: vec2<f32>) -> vec2<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(position_texture));
    let ratio = frame.upscale_ratio - 1.0;
    return uv + select(0.5, -0.5, (frame.number & 1u) == 0u) * texel_size * ratio;
}

// Normal-weighting function (4.4.1)
fn normal_weight(n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let exponent = 16.0;
    return pow(max(0.0, dot(n0, n1)), exponent);
}

// Depth-weighting function (4.4.2)
fn depth_weight(d0: f32, d1: f32, gradient: vec2<f32>, offset: vec2<f32>) -> f32 {
    let eps = 0.01;
    return exp((-abs(d0 - d1)) / (abs(dot(gradient, offset)) + eps));
}

// Luminance-weighting function (4.4.3)
fn luminance_weight(l0: f32, l1: f32, variance: f32) -> f32 {
    let strictness = 4.0;
    let exponent = 0.25;
    let eps = 0.001;
    return exp((-abs(l0 - l1)) / (strictness * pow(variance, exponent) + eps));
}

fn instance_weight(i0: f32, i1: f32) -> f32 {
    return max(0.0, 1.0 - abs(i0 - i1));
}

fn position_weight(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
    return exp(-distance(p0, p1));
}

fn load_input(coords: vec2<i32>) -> vec4<f32> {
#if DENOISE_LEVEL == 0
    return textureLoad(internal_texture_0, coords);
#else if DENOISE_LEVEL == 1
    return textureLoad(internal_texture_1, coords);
#else if DENOISE_LEVEL == 2
    return textureLoad(internal_texture_2, coords);
#else if DENOISE_LEVEL == 3
    return textureLoad(internal_texture_3, coords);
#endif
}

fn store_output(coords: vec2<i32>, value: vec4<f32>) {
#if DENOISE_LEVEL == 0
    textureStore(internal_texture_1, coords, value);
#else if DENOISE_LEVEL == 1
    textureStore(internal_texture_2, coords, value);
#else if DENOISE_LEVEL == 2
    textureStore(internal_texture_3, coords, value);
#else if DENOISE_LEVEL == 3
    textureStore(output_texture, coords, value);
#endif
}

fn step_size() -> i32 {
#if DENOISE_LEVEL == 0
    return 8;
#else if DENOISE_LEVEL == 1
    return 4;
#else if DENOISE_LEVEL == 2
    return 2;
#else if DENOISE_LEVEL == 3
    return 1;
#endif
}

fn accumulate_variance(
    uv: vec2<f32>,
    input_size: vec2<i32>,
    offset: vec2<i32>,
    sum_variance: ptr<function, f32>,
) {
    let sample_uv = uv + vec2<f32>(offset) / vec2<f32>(input_size);
    if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
        return;
    }

    let variance = textureSampleLevel(variance_texture, nearest_sampler, sample_uv, 0.0).x;
    if variance > F32_MAX {
        return;
    }

    *sum_variance += frame.kernel[offset.y + 1][offset.x + 1] * max(variance, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn demodulation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let input_size = textureDimensions(render_texture);
    let output_size = textureDimensions(output_texture);
    let deferred_size = textureDimensions(position_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, output_size);
    let deferred_uv = jittered_deferred_uv(uv);

    let albedo = textureSampleLevel(albedo_texture, nearest_sampler, deferred_uv, 0.0).rgb;
    var irradiance = textureSampleLevel(render_texture, nearest_sampler, uv, 0.0).rgb;
    irradiance = select(irradiance / albedo, vec3<f32>(0.0), albedo < vec3<f32>(0.01));

    let color = vec4<f32>(irradiance, 1.0);
    textureStore(internal_texture_0, coords, color);

    var sum_variance = 0.0;
    accumulate_variance(uv, input_size, vec2<i32>(-1, -1), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(-1, 0), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(-1, 1), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(0, -1), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(0, 0), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(0, 1), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(1, -1), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(1, 0), &sum_variance);
    accumulate_variance(uv, input_size, vec2<i32>(1, 1), &sum_variance);
    textureStore(internal_variance, coords, vec4<f32>(sum_variance));
}

fn accumulate_irradiance(
    output_size: vec2<i32>,
    coords: vec2<i32>,
    offset: vec2<i32>,
    normal: vec3<f32>,
    depth: f32,
    depth_gradient: vec2<f32>,
    instance: f32,
    lum: f32,
    variance: f32,
    sum_irradiance: ptr<function, vec3<f32>>,
    sum_w: ptr<function, f32>,
#ifdef FIREFLY_FILTERING
    ff_moment_1: ptr<function, f32>,
    ff_moment_2: ptr<function, f32>,
    ff_count: ptr<function, f32>,
#endif
) {
    let sample_coords = coords + offset * step_size();
    let sample_uv = coords_to_uv(sample_coords, output_size);
    let sample_deferred_uv = jittered_deferred_uv(sample_uv);
    if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
        return;
    }

    let irradiance = load_input(sample_coords).rgb;
    if any_is_nan_vec3(irradiance) || any(irradiance > vec3<f32>(F32_MAX)) {
        return;
    }

    let sample_normal = normalize(textureSampleLevel(normal_texture, nearest_sampler, sample_deferred_uv, 0.0).xyz);
    let sample_depth = textureSampleLevel(position_texture, nearest_sampler, sample_deferred_uv, 0.0).w;
    let sample_instance = textureSampleLevel(instance_material_texture, nearest_sampler, sample_deferred_uv, 0.0).x;
    let sample_luminance = luminance(irradiance);

    let w_normal = normal_weight(normal, sample_normal);
    let w_depth = depth_weight(depth, sample_depth, depth_gradient, vec2<f32>(offset));
    let w_instance = instance_weight(instance, sample_instance);
    let w_luminance = luminance_weight(lum, sample_luminance, variance);

    let w = clamp(w_normal * w_depth * w_instance * w_luminance, 0.0, 1.0) * frame.kernel[offset.y + 1][offset.x + 1];
    *sum_irradiance += irradiance * w;
    *sum_w += w;

#ifdef FIREFLY_FILTERING
    *ff_moment_1 += sample_luminance;
    *ff_moment_2 += sample_luminance * sample_luminance;
    *ff_count += 1.0;
#endif
}

@compute @workgroup_size(8, 8, 1)
fn denoise(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let input_size = textureDimensions(render_texture);
    let output_size = textureDimensions(output_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, output_size);
    let deferred_uv = jittered_deferred_uv(uv);

    let depth = textureSampleLevel(position_texture, nearest_sampler, deferred_uv, 0.0).w;
    let depth_gradient = textureSampleLevel(depth_gradient_texture, nearest_sampler, deferred_uv, 0.0).xy;
    let normal = normalize(textureSampleLevel(normal_texture, nearest_sampler, deferred_uv, 0.0).xyz);
    let instance = textureSampleLevel(instance_material_texture, nearest_sampler, deferred_uv, 0.0).x;

    if depth < F32_EPSILON {
        store_output(coords, vec4<f32>(0.0));
        return;
    }

    let variance = textureLoad(internal_variance, coords).x;
    var irradiance = load_input(coords).rgb;

    var sum_irradiance = irradiance * frame.kernel[1][1];
    var sum_w = frame.kernel[1][1];

    if any_is_nan_vec3(irradiance) || any(irradiance > vec3<f32>(F32_MAX)) {
        irradiance = vec3<f32>(0.0);
        sum_irradiance = vec3<f32>(0.0);
        sum_w = 0.0;
    }

    let lum = luminance(irradiance);

#ifdef FIREFLY_FILTERING
    var ff_moment_1 = 0.0;
    var ff_moment_2 = 0.0;
    var ff_count = 0.0;

    accumulate_irradiance(output_size, coords, vec2<i32>(-1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(0, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(-1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(-1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(0, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
#else
    accumulate_irradiance(output_size, coords, vec2<i32>(-1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(0, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(-1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(-1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(0, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
    accumulate_irradiance(output_size, coords, vec2<i32>(1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_irradiance, &sum_w);
#endif

    irradiance = select(sum_irradiance / sum_w, vec3<f32>(0.0), sum_w < 0.0001);

#ifdef FIREFLY_FILTERING
    let ff_mean = ff_moment_1 / ff_count;
    let ff_var = ff_moment_2 / ff_count - ff_mean * ff_mean;

    if lum > ff_mean + 3.0 * sqrt(ff_var) {
        irradiance = ff_mean / lum * irradiance;
    }
#endif

    var color = vec4<f32>(irradiance, 1.0);

#if DENOISE_LEVEL == 3
    // let velocity = textureLoad(velocity_uv_texture, deferred_coords, 0).xy;
    // let previous_uv = uv - velocity;
    // var previous_color = color;

    // for (var i = 0; i < 1; i += 1) {
    //     let sample_uv = previous_uv;

    //     let previous_depths = textureGather(3, previous_position_texture, linear_sampler, previous_uv);
    //     let previous_depth = max(max(previous_depths.x, previous_depths.y), max(previous_depths.z, previous_depths.w));
    //     if previous_depth == 0.0 {
    //         continue;
    //     }
    //     let depth_ratio = depth / previous_depth;
    //     let depth_miss = depth_ratio < 0.95 || depth_ratio > 1.05;

    //     let previous_velocity = textureSampleLevel(previous_velocity_uv_texture, nearest_sampler, previous_uv, 0.0).xy;
    //     let velocity_miss = distance(velocity, previous_velocity) > 0.0001;

    //     if depth_miss && velocity_miss {
    //         continue;
    //     }

    //     previous_color = textureSampleLevel(previous_radiance_texture, nearest_sampler, sample_uv, 0.0);
    // }

    // let mixed_color = mix(color, previous_color, 0.5);
    // color = select(mixed_color, color, any_is_nan_vec4(mixed_color) || previous_color.a == 0.0);
    // textureStore(radiance_texture, coords, color);

    let albedo = textureSampleLevel(albedo_texture, nearest_sampler, deferred_uv, 0.0);
    color *= albedo;
#endif

    store_output(coords, color);
}
