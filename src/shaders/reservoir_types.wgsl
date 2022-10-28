#define_import_path bevy_hikari::reservoir_types

// 64 Bytes
struct PackedReservoir {
    radiance: vec2<u32>,            // RGBA16F
    random: vec2<u32>,              // RGBA16F
    visible_position: vec4<f32>,    // RGBA32F
    sample_position: vec4<f32>,     // RGBA32F
    visible_normal: u32,            // RGBA8SN
    sample_normal: u32,             // RGBA8SN
    reservoir: vec2<u32>,           // RGBA16F
};

struct Reservoirs {
    data: array<PackedReservoir>,
};

struct Sample {
    radiance: vec4<f32>,
    random: vec4<f32>,
    visible_position: vec4<f32>,
    visible_normal: vec3<f32>,
    visible_instance: u32,
    sample_position: vec4<f32>,
    sample_normal: vec3<f32>,
};

struct Reservoir {
    s: Sample,
    count: f32,
    w: f32,
    w_sum: f32,
    w2_sum: f32,
};
