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

[[group(1), binding(0)]]
var<uniform> mesh: Mesh;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
    projection: mat4x4<f32>;
};

struct List {
    data: array<u32, 4194304>;
    counter: atomic<u32>;
};

struct Node {
    children: u32;
};

struct Octree {
    nodes: array<Node, 524288>;
    levels: array<u32, 8>;
    node_counter: atomic<u32>;
    level_counter: atomic<u32>;
};

[[group(2), binding(0)]]
var<uniform> volume: Volume;

[[group(2), binding(1)]]
var<storage, read_write> fragments: List;

[[group(2), binding(2)]]
var<storage, read_write> octree: Octree;

[[group(2), binding(3)]]
var texture: texture_storage_1d<rgba8unorm, read_write>;

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_position = mesh.model * vec4<f32>(vertex.position, 1.0);
    let view_normal = volume.projection * mesh.model * vec4<f32>(vertex.normal, 0.0);

    var out: VertexOutput;
    out.clip_position = volume.projection * world_position;
    out.world_position = world_position;
    out.view_normal = view_normal.xyz;
    return out;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal_proj = dot(in.view_normal, vec3<f32>(0.0, 0.0, 1.0));
    let normal_cond = abs(normal_proj) > 0.707;

    if (normal_cond) {
        let id = atomicAdd(&fragments.counter, 1u);
        let position = (in.world_position.xyz - volume.min) / (volume.max - volume.min);
        fragments.data[id] = pack4x8unorm(vec4<f32>(position, 1.0));
    }

    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}