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
var internal_variance: texture_storage_2d<r32float, read_write>;

@group(4) @binding(0)
var albedo_texture: texture_2d<f32>;
@group(4) @binding(1)
var variance_texture: texture_2d<f32>;
@group(4) @binding(2)
var previous_render_texture: texture_2d<f32>;
@group(4) @binding(3)
var render_texture: texture_2d<f32>;
@group(4) @binding(4)
var output_texture: texture_storage_2d<rgba16float, read_write>;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 0xFFFFFFFFu;

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Normal-weighting function (4.4.1)
fn normal_weight(n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let exponent = 16.0;
    return pow(max(0.0, dot(n0, n1)), exponent);
}

// Depth-weighting function (4.4.2)
fn depth_weight(d0: f32, d1: f32, gradient: vec2<f32>, offset: vec2<i32>) -> f32 {
    let eps = 0.01;
    return exp((-abs(d0 - d1)) / (abs(dot(gradient, vec2<f32>(offset))) + eps));
}

// Luminance-weighting function (4.4.3)
fn luminance_weight(l0: f32, l1: f32, variance: f32) -> f32 {
    let strictness = 64.0;
    let eps = 0.001;
    return exp((-abs(l0 - l1)) / (strictness * sqrt(variance) + eps));
}

fn instance_weight(i0: u32, i1: u32) -> f32 {
    return f32(i0 == i1);
}

fn load_input(coords: vec2<i32>) -> vec4<f32> {
#ifdef DENOISE_LEVEL_0
    return textureLoad(render_texture, coords, 0);
#endif
#ifdef DENOISE_LEVEL_1
    return textureLoad(internal_texture_0, coords);
#endif
#ifdef DENOISE_LEVEL_2
    return textureLoad(internal_texture_1, coords);
#endif
#ifdef DENOISE_LEVEL_3
    return textureLoad(internal_texture_2, coords);
#endif
}

fn load_irradiance(coords: vec2<i32>) -> vec3<f32> {
    var irradiance = load_input(coords).rgb;
#ifdef DENOISE_LEVEL_0
    let albedo = textureLoad(albedo_texture, coords, 0).rgb;
    irradiance = select(irradiance / albedo, vec3<f32>(0.0), albedo < vec3<f32>(0.01));
#endif
    return irradiance;
}

fn store_output(coords: vec2<i32>, value: vec4<f32>) {
#ifdef DENOISE_LEVEL_0
    textureStore(internal_texture_0, coords, value);
#endif
#ifdef DENOISE_LEVEL_1
    textureStore(internal_texture_1, coords, value);
#endif
#ifdef DENOISE_LEVEL_2
    textureStore(internal_texture_2, coords, value);
#endif
#ifdef DENOISE_LEVEL_3
    textureStore(output_texture, coords, value);
#endif
}

fn load_cache_variance(coords: vec2<i32>, size: vec2<i32>) -> f32 {
    var sum_variance = 0.0;
#ifdef DENOISE_LEVEL_0
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = coords + offset;
            if any(sample_coords < vec2<i32>(0)) || any(sample_coords >= size) {
                continue;
            }

            let variance = textureLoad(variance_texture, sample_coords, 0).x;
            if variance > F32_MAX {
                continue;
            }

            sum_variance += frame.kernel[y + 1][x + 1] * variance;
        }
    }
    sum_variance = max(sum_variance, 0.0);
    textureStore(internal_variance, coords, vec4<f32>(sum_variance));
#else
    sum_variance = textureLoad(internal_variance, coords).x;
#endif
    return sum_variance;
}

fn step_size() -> i32 {
#ifdef DENOISE_LEVEL_0
    return 8;
#endif
#ifdef DENOISE_LEVEL_1
    return 4;
#endif
#ifdef DENOISE_LEVEL_2
    return 2;
#endif
#ifdef DENOISE_LEVEL_3
    return 1;
#endif
}

@compute @workgroup_size(8, 8, 1)
fn denoise(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(output_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, size);

    let deferred_size = textureDimensions(position_texture);
    let deferred_coords = vec2<i32>(uv * vec2<f32>(deferred_size));

    let position_depth = textureSampleLevel(position_texture, nearest_sampler, uv, 0.0);
    let depth_gradient = textureSampleLevel(depth_gradient_texture, nearest_sampler, uv, 0.0).xy;
    let normal = normalize(textureSampleLevel(normal_texture, nearest_sampler, uv, 0.0).xyz);
    let instance = textureLoad(instance_material_texture, deferred_coords, 0).x;

    if position_depth.w < F32_EPSILON {
        store_output(coords, vec4<f32>(0.0));
        return;
    }

    let variance = load_cache_variance(coords, size);

    var irradiance = load_irradiance(coords);
    irradiance = select(irradiance, vec3<f32>(0.0), irradiance > vec3<f32>(F32_MAX));
    let lum = luminance(irradiance);

    var sum_irradiance = vec3<f32>(0.0);
    var sum_w = 0.0;

    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = coords + offset * step_size();
            if any(sample_coords < vec2<i32>(0)) || any(sample_coords >= size) {
                continue;
            }

            irradiance = load_irradiance(sample_coords);
            irradiance = select(irradiance, vec3<f32>(0.0), irradiance > vec3<f32>(F32_MAX));

            let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
            let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
            let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
            let sample_luminance = luminance(irradiance);

            let w_normal = normal_weight(normal, sample_normal);
            let w_depth = depth_weight(position_depth.w, sample_depth, depth_gradient, offset);
            let w_instance = instance_weight(instance, sample_instance);
            let w_luminance = luminance_weight(lum, sample_luminance, variance);

            let w = clamp(w_normal * w_depth * w_instance * w_luminance, 0.0, 1.0) * frame.kernel[y + 1][x + 1];
            sum_irradiance += irradiance * w;
            sum_w += w;
        }
    }

    irradiance = select(sum_irradiance / sum_w, vec3<f32>(0.0), sum_w < 0.0001);
    var color = vec4<f32>(irradiance, 1.0);

#ifdef DENOISE_LEVEL_3
    let albedo = textureLoad(albedo_texture, coords, 0);
    color *= albedo;

    let velocity = textureSampleLevel(velocity_uv_texture, nearest_sampler, uv, 0.0).xy;
    let previous_uv = uv - velocity;
    // var previous_color = textureSampleLevel(previous_render_texture, linear_sampler, previous_uv, 0.0);
    var previous_color = color;

    var min_distance = F32_MAX;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let uv_offset = vec2<f32>(offset) / vec2<f32>(deferred_size);
            let sample_uv = previous_uv + uv_offset;

            let sample_coords = vec2<i32>(sample_uv * vec2<f32>(deferred_size));
            let previous_instance = textureLoad(previous_instance_material_texture, sample_coords, 0).x;
            if (instance != previous_instance) {
                continue;
            }

            let previous_position_depth = textureSampleLevel(previous_position_texture, nearest_sampler, sample_uv, 0.0);
            if previous_position_depth.w == 0.0 {
                continue;
            }
            let depth_ratio = position_depth.w / previous_position_depth.w;
            if depth_ratio < 0.9 || depth_ratio > 1.1 {
                continue;
            }

            let previous_normal = normalize(textureSampleLevel(previous_normal_texture, nearest_sampler, sample_uv, 0.0).xyz);
            if dot(normal, previous_normal) < 0.866 {
                continue;
            }

            let dist = distance(previous_position_depth.xyz, position_depth.xyz);
            if (dist < min_distance) {
                min_distance = dist;
                previous_color = textureSampleLevel(previous_render_texture, linear_sampler, sample_uv, 0.0);
            }
        }
    }

    let mix_rate = select(0.0, 0.8, previous_color.a > 0.0);
    color = mix(color, previous_color, mix_rate);
#endif

    store_output(coords, color);
}
