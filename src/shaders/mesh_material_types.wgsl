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

struct MeshIndex {
    vertex: u32,
    primitive: u32,
    node: vec2<u32>,    // x: offset, y: size
};

struct Instance {
    min: vec3<f32>,
    material: u32,
    max: vec3<f32>,
    node_index: u32,
    model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
    mesh: MeshIndex,
};

struct Node {
    min: vec3<f32>,
    entry_index: u32,
    max: vec3<f32>,
    exit_index: u32,
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

struct AliasEntry {
    prob: f32,
    index: u32,
}

struct Emissive {
    emissive: vec4<f32>,
    position: vec3<f32>,
    radius: f32,
    instance: u32,
    alias_table: vec2<u32>,
    surface_area: f32,
};

type Vertices = array<Vertex>;
type Primitives = array<Primitive>;
type Instances = array<Instance>;
type Materials = array<Material>;
type AliasTable = array<AliasEntry>;

struct Nodes {
    count: u32,
    data: array<Node>,
};

struct Emissives {
    count: u32,
    data: array<Emissive>,
};
