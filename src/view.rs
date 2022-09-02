use crate::transform::PreviousGlobalTransform;
use bevy::{
    prelude::*,
    render::{
        render_resource::{DynamicUniformBuffer, ShaderType},
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
        RenderApp, RenderStage,
    },
};

pub struct ViewPlugin;
impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<PreviousViewUniforms>()
                .add_system_to_stage(RenderStage::Prepare, prepare_view_uniforms);
        }
    }
}

#[derive(Clone, ShaderType)]
pub struct PreviousViewUniform {
    view_proj: Mat4,
    inverse_view_proj: Mat4,
}

#[derive(Default)]
pub struct PreviousViewUniforms {
    pub uniforms: DynamicUniformBuffer<PreviousViewUniform>,
}

#[derive(Component)]
pub struct PreviousViewUniformOffset {
    pub offset: u32,
}

fn prepare_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<PreviousViewUniforms>,
    views: Query<(Entity, &ExtractedView, &PreviousGlobalTransform)>,
) {
    view_uniforms.uniforms.clear();
    for (entity, camera, transform) in &views {
        let projection = camera.projection;
        let inverse_projection = projection.inverse();
        let view = transform.compute_matrix();
        let inverse_view = view.inverse();
        let view_uniforms = PreviousViewUniformOffset {
            offset: view_uniforms.uniforms.push(PreviousViewUniform {
                view_proj: projection * inverse_view,
                inverse_view_proj: view * inverse_projection,
            }),
        };

        commands.entity(entity).insert(view_uniforms);
    }

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}
