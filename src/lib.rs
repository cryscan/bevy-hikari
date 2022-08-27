use bevy::{asset::load_internal_asset, prelude::*, reflect::TypeUuid};
use image::ImagePlugin;
use mesh::BindlessMeshPlugin;
use prepass::PrepassPlugin;

pub mod image;
pub mod mesh;
pub mod prelude;
pub mod prepass;

pub const PREPASS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4693612430004931427);

pub struct HikariPlugin;
impl Plugin for HikariPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PREPASS_SHADER_HANDLE,
            "shaders/prepass.wgsl",
            Shader::from_wgsl
        );

        app.add_plugin(BindlessMeshPlugin)
            .add_plugin(ImagePlugin)
            .add_plugin(PrepassPlugin);
    }
}
