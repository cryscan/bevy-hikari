#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_types

@group(1) @binding(0)
var<uniform> mesh: Mesh;

#import bevy_pbr::mesh_functions

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let model = mesh.model;
    let vertex_position = vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.clip_position = mesh_position_local_to_clip(model, vertex_position);
    out.world_normal = mesh_normal_local_to_world(vertex.normal);

    return out;
}

struct FragmentOutput {
    @location(0) world_normal: vec2<f32>,
};

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.world_normal = in.world_normal.xy;
    return out;
}
