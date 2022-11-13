#import bevy_core_pipeline::tonemapping
#import bevy_hikari::utils

@group(0) @binding(0)
var input_texture: texture_2d<f32>;
@group(0) @binding(1)
var linear_sampler: sampler;

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

fn inverse_reintard_luminance(color: vec3<f32>) -> vec3<f32> {
    let l_old = tonemapping_luminance(color);
    let l_new = l_old / max(1.0 - l_old, 0.000015);
    return tonemapping_change_luminance(color, l_new);
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let uv = clip_to_uv(vec4<f32>(in.position, 1.0));

    out.color = textureSample(input_texture, linear_sampler, uv);

#ifdef HDR
    out.color = vec4<f32>(inverse_reintard_luminance(out.color.rgb), out_color.a);
#endif

    return out;
}