#import bevy_pbr::mesh_struct

struct Vertex {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] view_normal: vec3<f32>;
};

struct View {
    view_proj: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
};

[[group(0), binding(0)]]
var<uniform> view: View;

[[group(1), binding(0)]]
var<uniform> mesh: Mesh;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
};

struct List {
    data: array<u32, 1048576>;
    counter: atomic<u32>;
};

struct Node {
    children: u32;
};

struct Octree {
    nodes: array<Node, 131072>;
    levels: array<u32, 8>;
    node_counter: atomic<u32>;
    level_counter: atomic<u32>;
};

struct Radiance {
    data: array<u32, 2097152>;
};

[[group(2), binding(0)]]
var<uniform> volume: Volume;

[[group(2), binding(1)]]
var<storage, read_write> fragments: List;

[[group(2), binding(2)]]
var<storage, read_write> octree: Octree;

[[group(2), binding(3)]]
var<storage, read_write> radiance: Radiance;

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_position = mesh.model * vec4<f32>(vertex.position, 1.0);
    let view_normal = view.view_proj * mesh.model * vec4<f32>(vertex.normal, 0.0);

    var out: VertexOutput;
    out.clip_position = view.view_proj * world_position;
    out.world_position = world_position;
    out.view_normal = view_normal.xyz;
    return out;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal = abs(in.view_normal);

    if (normal.z >= max(normal.x, normal.y)) {
        let id = atomicAdd(&fragments.counter, 1u);
        let position = (in.world_position.xyz - volume.min) / (volume.max - volume.min);
        fragments.data[id] = pack4x8unorm(vec4<f32>(position, 1.0));
    }

    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}