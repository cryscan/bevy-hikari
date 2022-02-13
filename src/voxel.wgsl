#import bevy_pbr::mesh_struct
#import bevy_pbr::mesh_view_bind_group

struct Vertex {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
};

struct FragmentInput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
}

struct View {
    view_proj: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
};

[[group(0), binding(0)]]
var<uniform> view: View;

[[group(1), binding(0)]]
var<uniform> mesh: Mesh;

[[group(2), binding(1)]]
var voxel_texture: texture_storage_3d<rgba8unorm, read_write>;

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_position = mesh.model * vec4<f32>(vertex.position, 1.0);
    let world_normal = mesh.model * vec4<f32>(vertex.normal, 0.0);

    var out: VertexOutput;
    out.clip_position = view.view_proj * world_position;
    out.world_position = world_position;
    out.world_normal = world_normal.xyz;
    return out;
}

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}