#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

[[group(1), binding(0)]]
var irradiance_texture: texture_2d<f32>;
[[group(1), binding(1)]]
var irradiance_sampler: sampler;
[[group(1), binding(2)]]
var albedo_texture: texture_2d<f32>;
[[group(1), binding(3)]]
var albedo_sampler: sampler;

let POSITIONS: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(1., 1., 0.),
    vec3<f32>(1., -1., 0.),
    vec3<f32>(-1., -1., 0.),
    vec3<f32>(-1., 1., 0.),
);

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vertex([[builtin(vertex_index)]] id: u32) -> VertexOutput {
    var position: vec2<f32>;
    position.x = -2.0 * f32(id / 2u) + 1.0;
    position.y = -2.0 * f32(((id + 1u) % 4u) / 2u) + 1.0;

    var out: VertexOutput;
    out.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    out.clip_position = vec4<f32>(position, 0.1, 1.0);
    return out;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var irradiance = textureSample(irradiance_texture, irradiance_sampler, in.uv);
    var base_color = textureSample(albedo_texture, albedo_sampler, in.uv);
    return irradiance * base_color;
}