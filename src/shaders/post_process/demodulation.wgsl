#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings

#import bevy_core_pipeline::fullscreen_vertex_shader

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var render_texture: texture_2d<f32>;
@group(3) @binding(1)
var albedo_texture: texture_2d<f32>;
@group(3) @binding(2)
var variance_texture: texture_2d<f32>;

fn is_nan(val: f32) -> bool {
    return !(val < 0.0 || 0.0 < val || val == 0.0);
}

struct FragmentOutput {
    @location(0) 
    radiance: vec4<f32>,
    @location(1)
    variance: f32,
};

fn jittered_deferred_uv(uv: vec2<f32>, deferred_texel_size: vec2<f32>) -> vec2<f32> {
    return uv + select(0.5, -0.5, (frame.number & 1u) == 0u) * deferred_texel_size;
}

fn accumulate_variance(
    uv: vec2<f32>,
    texel_size: vec2<f32>,
    offset: vec2<i32>,
    sum_variance: ptr<function, f32>
) {
    let sample_uv = uv + vec2<f32>(offset) * texel_size;
    if any(abs(sample_uv - 0.5) > vec2<f32>(0.5)) {
        return;
    }

    let variance = textureSample(variance_texture, nearest_sampler, sample_uv).x;
    if !is_nan(variance) {
        *sum_variance += frame.kernel[offset.y + 1][offset.x + 1] * max(variance, 0.0);
    }
}

@fragment
fn demodulation(@location(0) uv: vec2<f32>) -> FragmentOutput {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(render_texture));
    let deferred_texel_size = 1.0 / view.viewport.zw;
    let deferred_uv = jittered_deferred_uv(uv, deferred_texel_size);

    var out: FragmentOutput;

    let albedo = textureSample(albedo_texture, nearest_sampler, deferred_uv);
    out.radiance = textureSample(render_texture, nearest_sampler, uv);
    out.radiance = select(vec4<f32>(0.0), out.radiance / albedo, albedo > vec4<f32>(0.0));

    var variance = 0.0;
    accumulate_variance(uv, texel_size, vec2<i32>(-1, -1), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(-1, 0), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(-1, 1), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(0, -1), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(0, 0), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(0, 1), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(1, -1), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(1, 0), &variance);
    accumulate_variance(uv, texel_size, vec2<i32>(1, 1), &variance);
    out.variance = variance;

    return out;
}