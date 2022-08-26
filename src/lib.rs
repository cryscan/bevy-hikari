use bevy::prelude::*;
use mesh::BindlessMeshPlugin;
use prepass::PrepassPlugin;

pub mod mesh;
pub mod prelude;
pub mod prepass;

pub struct HikariPlugin;
impl Plugin for HikariPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(BindlessMeshPlugin).add_plugin(PrepassPlugin);
    }
}
