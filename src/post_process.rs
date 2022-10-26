use bevy::{
    prelude::*,
    render::{render_resource::*, RenderApp},
};

pub struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(_render_app) = app.get_sub_app_mut(RenderApp) {}
    }
}

pub struct PostProcessPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
}
