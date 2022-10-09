#define_import_path bevy_hikari::deferred_bindings

@group(1) @binding(0)
var position_texture: texture_2d<f32>;
@group(1) @binding(1)
var normal_texture: texture_2d<f32>;
@group(1) @binding(2)
var depth_gradient_texture: texture_2d<f32>;
@group(1) @binding(3)
var uv_texture: texture_2d<f32>;
@group(1) @binding(4)
var velocity_texture: texture_2d<f32>;
@group(1) @binding(5)
var instance_material_texture: texture_2d<u32>;
@group(1) @binding(6)
var albedo_texture: texture_storage_2d<rgba16float, read_write>;