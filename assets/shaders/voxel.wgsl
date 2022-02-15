#import bevy_pbr::mesh_struct
#import bevy_pbr::mesh_view_bind_group

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
};

struct FragmentList {
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
var<storage, read_write> fragments: FragmentList;

[[group(2), binding(2)]]
var<storage, read_write> octree: Octree;

[[group(2), binding(3)]]
var texture: texture_storage_1d<rgba8unorm, read_write>;

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
    let position = in.world_position.xyz;
    let min_cond = all(position >= volume.min);
    let max_cond = all(position <= volume.max);

    let normal_proj = dot(in.view_normal, vec3<f32>(0.0, 0.0, 1.0));
    let normal_cond = abs(normal_proj) > 0.707;

    if (min_cond && max_cond && normal_cond) {
        let id = atomicAdd(&fragments.counter, 1u);
        let unit_position = (position - volume.min) / (volume.max - volume.min);
        fragments.data[id] = pack4x8unorm(vec4<f32>(unit_position, 1.0));
    }

    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}