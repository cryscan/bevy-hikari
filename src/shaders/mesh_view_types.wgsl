#define_import_path bevy_hikari::mesh_view_types

struct Frame {
    kernel: mat3x3<f32>,
    clear_color: vec4<f32>,
    number: u32,
    direct_validate_interval: u32,
    emissive_validate_interval: u32,
    indirect_bounces: u32,
    suppress_temporal_reuse: u32,
    max_temporal_reuse_count: u32,
    max_spatial_reuse_count: u32,
    solar_angle: f32,
    max_indirect_luminance: f32,
};

struct PreviousView {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
};

struct PreviousMesh {
    model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
};

struct InstanceIndex {
    instance: u32,
    material: u32
};