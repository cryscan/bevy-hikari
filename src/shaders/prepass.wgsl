#import bevy_pbr::mesh_view_types
#import bevy_pbr::mesh_types

struct PreviousView {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
};

struct PreviousMesh {
    model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> previous_view: PreviousView;

@group(1) @binding(0)
var<uniform> mesh: Mesh;
@group(1) @binding(1)
var<uniform> previous_mesh: PreviousMesh;

#import bevy_pbr::mesh_functions

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) velocity: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let model = mesh.model;
    let vertex_position = vec4<f32>(vertex.position, 1.0);    
    
    var out: VertexOutput;
    out.clip_position = mesh_position_local_to_clip(model, vertex_position);
    out.world_normal = mesh_normal_local_to_world(vertex.normal);

    let previous_world_position = mesh_position_local_to_world(previous_mesh.model, vertex_position);
    let previous_clip_position = previous_view.view_proj * previous_world_position;

    out.velocity = out.clip_position.xy / out.clip_position.w;
    out.velocity = out.velocity - previous_clip_position.xy / previous_clip_position.w;
    out.velocity = out.velocity / 2.0;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4<f32>(in.world_normal.xy, in.velocity);
    return out;
}
