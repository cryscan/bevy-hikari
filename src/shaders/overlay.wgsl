#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

[[group(1), binding(0)]]
var texture: texture_2d<f32>;
[[group(1), binding(1)]]
var textuer_sampler: sampler;

let POSITIONS: mat3x4<f32> = mat3x4<f32>(
    vec3<f32>(-1., -1., 0.),
    vec3<f32>(1., -1., 0.),
    vec3<f32>(1., 1., 0.),
    vec3<f32>(-1., 1., 0.),
);

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vertex([[builtin(vertex_index)]] id: u32) -> VertexOutput {
    let position = POSITIONS[id];

    var out: VertexOutput;
    out.uv = (position.xy + vec2<f32>(1., 1.)) * vec2<f32>(0.5, -0.5);
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(texture, textuer_sampler, in.uv);
}