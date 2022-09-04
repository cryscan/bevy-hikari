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
    @location(0) world_position: vec4<f32>,
    @location(1) previous_world_position: vec4<f32>,
    @location(2) world_normal: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let model = mesh.model;
    let vertex_position = vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.world_position = mesh_position_local_to_world(model, vertex_position);
    out.previous_world_position = mesh_position_local_to_world(previous_mesh.model, vertex_position);
    out.world_normal = mesh_normal_local_to_world(vertex.normal);
    out.clip_position = view.view_proj * out.world_position;

    return out;
}

struct FragmentInput {
    @location(0) world_position: vec4<f32>,
    @location(1) previous_world_position: vec4<f32>,
    @location(2) world_normal: vec3<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    return clip.xy / (2.0 * clip.w);
}

@fragment
fn fragment(in: FragmentInput) -> FragmentOutput {
    let clip_position = view.view_proj * in.world_position;
    let previous_clip_position = previous_view.view_proj * in.previous_world_position;
    let velocity = clip_to_uv(clip_position) - clip_to_uv(previous_clip_position);

    var out: FragmentOutput;
    out.color = vec4<f32>(in.world_normal.xy, velocity);
    return out;
}
