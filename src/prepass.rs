use bevy::prelude::*;

pub struct PrepassPlugin;
impl Plugin for PrepassPlugin {
    fn build(&self, _app: &mut App) {}
}

pub struct PrepassTextures {
    pub position: Handle<Image>,
    pub normal: Handle<Image>,
}
