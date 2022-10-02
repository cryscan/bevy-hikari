#define_import_path bevy_hikari::mesh_material_types

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
    material: u32,
    node_index: u32,
};

struct Node {
    min: vec3<f32>,
    max: vec3<f32>,
    entry_index: u32,
    exit_index: u32,
    primitive_index: u32,
};

struct Material {
    base_color: vec4<f32>,
    base_color_texture: u32,

    emissive: vec4<f32>,
    emissive_texture: u32,

    perceptual_roughness: f32,
    metallic: f32,
    metallic_roughness_texture: u32,
    reflectance: f32,

    normal_map_texture: u32,
    occlusion_texture: u32,
};

struct LightSource {
    emissive: vec4<f32>,
    position: vec3<f32>,
    radius: f32,
    instance: u32,
};

struct Vertices {
    data: array<Vertex>,
};
struct Primitives {
    data: array<Primitive>,
};
struct Instances {
    data: array<Instance>,
};
struct Nodes {
    count: u32,
    data: array<Node>,
};
struct Materials {
    data: array<Material>,
}

struct LightSources {
    count: u32,
    data: array<LightSource>,
};