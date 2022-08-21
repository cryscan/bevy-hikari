use bevy::prelude::*;

pub mod mesh;
pub mod prelude;

pub struct GiPlugin;

impl Plugin for GiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(mesh::BatchMeshPlugin);
    }
}
