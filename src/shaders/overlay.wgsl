#import bevy_hikari::utils

@group(0) @binding(0)
var input_texture: texture_storage_2d<rgba8unorm, read_write>;

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
    let coords = vec2<i32>(in.clip_position.xy);

    out.color = textureLoad(input_texture, coords);
    return out;
}