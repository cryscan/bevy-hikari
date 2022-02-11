#import bevy_pbr::mesh_struct

struct Vertex {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
#endif
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
};

struct FragmentInput {
    [[location(0)]] clip_position: vec4<f32>;
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

[[group(2), binding(0)]]
var fragments: texture_storage_1d<rgba8unorm, read_write>;

[[group(2), binding(1)]]
var<storage> fragments_counter: atomic<u32>;

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view.view_proj * mesh.model * vec4<f32>(vertex.position, 1.0);
    return out;
}

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    let id: i32 = (i32)atomicAdd(&fragments_counter, 1u);
    
    let position = in.clip_position;
    position.xy = position.xy / 2.0 + 0.5;

    textureStore(fragments, id, position);

    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}