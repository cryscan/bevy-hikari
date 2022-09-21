@group(0) @binding(0)
var render_texture: texture_2d<f32>;
@group(0) @binding(1)
var render_sampler: sampler;

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

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = 0.5 * in.position.xy + 0.5;
    uv.y = 1.0 - uv.y;
    let color = textureSample(render_texture, render_sampler, uv);
    return color;
}