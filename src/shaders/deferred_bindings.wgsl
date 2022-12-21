#define_import_path bevy_hikari::deferred_bindings

@group(1) @binding(0)
var position_texture: texture_2d<f32>;
@group(1) @binding(1)
var normal_texture: texture_2d<f32>;
@group(1) @binding(2)
var depth_gradient_texture: texture_2d<f32>;
@group(1) @binding(3)
var instance_material_texture: texture_2d<f32>;
@group(1) @binding(4)
var velocity_uv_texture: texture_2d<f32>;
@group(1) @binding(5)
var previous_position_texture: texture_2d<f32>;
@group(1) @binding(6)
var previous_normal_texture: texture_2d<f32>;
@group(1) @binding(7)
var previous_instance_material_texture: texture_2d<u32>;
@group(1) @binding(8)
var previous_velocity_uv_texture: texture_2d<f32>;