#define_import_path bevy_hikari::ray_tracing_types

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
};

struct Primitive {
    vertices: array<vec3<f32>, 3>,
    indices: array<u32, 3>,
    node_index: u32,
};

struct Slice {
    vertex: u32,
    primitive: u32,
    node_offset: u32,
    node_len: u32,
};

struct Instance {
    min: vec3<f32>,
    max: vec3<f32>,
    model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
    slice: Slice,
    node_index: u32,
};

struct Node {
    min: vec3<f32>,
    max: vec3<f32>,
    entry_index: u32,
    exit_index: u32,
    primitive_index: u32,
};

struct Vertices {
    data: array<Vertex>;
};
struct Primitives {
    data: array<Primitive>;
};
struct Instances {
    data: array<Instance>;
};
struct Nodes {
    data: array<Node>;
};

struct Ray {
    origin: vec3<f32>,
    inv_direction: vec3<f32>,
    signs: u32,
};