#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var radiance_texture: texture_2d<f32>;
@group(3) @binding(1)
var variance_texture: texture_2d<f32>;

fn jittered_deferred_uv(uv: vec2<f32>, deferred_texel_size: vec2<f32>) -> vec2<f32> {
    return uv + select(0.5, -0.5, (frame.number & 1u) == 0u) * deferred_texel_size;
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

fn instance_weight(i0: u32, i1: u32) -> f32 {
    return f32(i0 == i1);
}

fn stride() -> i32 {
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

fn accumulate_radiance(
    uv: vec2<f32>,
    deferred_uv: vec2<f32>,
    texel_size: vec2<f32>,
    offset: vec2<i32>,
    normal: vec3<f32>,
    depth: f32,
    depth_gradient: vec2<f32>,
    instance: u32,
    lum: f32,
    variance: f32,
    sum_radiance: ptr<function, vec3<f32>>,
    sum_w: ptr<function, f32>,
#ifdef FIREFLY_FILTERING
    ff_moment_1: ptr<function, f32>,
    ff_moment_2: ptr<function, f32>,
    ff_count: ptr<function, f32>,
#endif
) {
    let sample_uv = uv + texel_size * vec2<f32>(offset * stride());
    let sample_deferred_uv = deferred_uv + texel_size * vec2<f32>(offset * stride());

    if any(abs(sample_uv - 0.5) > vec2<f32>(0.5)) {
        return;
    }

    let radiance = textureSample(radiance_texture, nearest_sampler, sample_uv).rgb;
    if any_is_nan_vec3(radiance) {
        return;
    }

    let sample_normal = textureSample(normal_texture, nearest_sampler, sample_deferred_uv).xyz;
    let sample_depth = textureSample(position_texture, nearest_sampler, sample_deferred_uv).w;
    let sample_instance = u32(textureSample(instance_material_texture, nearest_sampler, sample_deferred_uv).x);
    let sample_lum = luminance(radiance);

    let w_normal = normal_weight(normal, sample_normal);
    let w_depth = depth_weight(depth, sample_depth, depth_gradient, vec2<f32>(offset));
    let w_instance = instance_weight(instance, sample_instance);
    let w_luminance = luminance_weight(lum, sample_lum, variance);

    let w = clamp(w_normal * w_depth * w_instance * w_luminance, 0.0, 1.0) * frame.kernel[offset.y + 1][offset.x + 1];
    *sum_radiance += radiance * w;
    *sum_w += w;

#ifdef FIREFLY_FILTERING
    *ff_moment_1 += sample_lum;
    *ff_moment_2 += sample_lum * sample_lum;
    *ff_count += 1.0;
#endif
}

@fragment
fn denoise(uv: vec2<f32>) -> @location(0) vec4<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(radiance_texture));
    let deferred_texel_size = 1.0 / view.viewport.zw;
    let deferred_uv = jittered_deferred_uv(uv, deferred_texel_size);

    let depth = textureSample(position_texture, nearest_sampler, deferred_uv).w;
    let depth_gradient = textureSample(depth_gradient_texture, nearest_sampler, deferred_uv).xy;
    let normal = textureSample(normal_texture, nearest_sampler, deferred_uv).xyz;
    let instance = u32(textureSample(normal_texture, nearest_sampler, deferred_uv).x);

    if depth == 0.0 {
        return vec4<f32>(0.0);
    }

    let variance = textureSample(variance_texture, nearest_sampler, uv).x;
    let radiance = textureSample(radiance_texture, nearest_sampler, uv);
    let lum = luminance(radiance.rgb);

    var sum_radiance = radiance.rgb * frame.kernel[1][1];
    var sum_w = frame.kernel[1][1];

    if any_is_nan_vec3(radiance.rgb) {
        sum_radiance = vec3<f32>(0.0);
        sum_w = 0.0;
    }

#ifdef FIREFLY_FILTERING
    var ff_moment_1 = 0.0;
    var ff_moment_2 = 0.0;
    var ff_count = 0.0;

    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(0, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(0, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w, &ff_moment_1, &ff_moment_2, &ff_count);
#else
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(-1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(0, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(0, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, -1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, 0), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
    accumulate_radiance(uv, deferred_uv, texel_size, vec2<i32>(1, 1), normal, depth, depth_gradient, instance, lum, variance, &sum_radiance, &sum_w);
#endif

    var color = select(vec3<f32>(0.0), sum_radiance / sum_w, sum_w > 0.0);

#ifdef FIREFLY_FILTERING
    let ff_mean = ff_monent_1 / ff_count;
    let ff_var = ff_moment_2 / ff_count - ff_mean * ff_mean;

    if lum > ff_mean + 3.0 * sqrt(ff_var) {
        color = ff_mean / lum * color;
    }
#endif

    return vec4<f32>(color, 1.0);
}