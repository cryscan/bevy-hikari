use crate::transform::PreviousGlobalTransform;
use bevy::{
    prelude::*,
    render::{
        render_resource::*,
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
                .init_resource::<FrameCounter>()
                .init_resource::<FrameUniform>()
                .add_system_to_stage(RenderStage::Prepare, prepare_view_uniforms)
                .add_system_to_stage(RenderStage::Prepare, prepare_frame_uniform);
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

#[derive(Default)]
pub struct FrameCounter(pub usize);

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuFrame {
    pub number: u32,
    pub kernel: Mat3,
}

#[derive(Default)]
pub struct FrameUniform {
    pub buffer: UniformBuffer<GpuFrame>,
}

fn prepare_frame_uniform(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut uniform: ResMut<FrameUniform>,
    mut counter: ResMut<FrameCounter>,
) {
    // let mut kernel = [Vec3::ZERO; 25];
    // for i in 0..5 {
    //     for j in 0..5 {
    //         let offset = IVec2::new(i - 2, j - 2);
    //         let index = (i + 5 * j) as usize;
    //         let value = match (offset.x.abs(), offset.y.abs()) {
    //             (0, 0) => 9.0 / 64.0,
    //             (0, 1) | (1, 0) => 3.0 / 32.0,
    //             (1, 1) => 1.0 / 16.0,
    //             (0, 2) | (2, 0) => 3.0 / 128.0,
    //             (1, 2) | (2, 1) => 1.0 / 64.0,
    //             (2, 2) => 1.0 / 256.0,
    //             _ => 0.0,
    //         };
    //         kernel[index] = Vec3::new(offset.x as f32, offset.y as f32, value);
    //     }
    // }

    uniform.buffer.set(GpuFrame {
        number: counter.0 as u32,
        kernel: Mat3 {
            x_axis: Vec3::new(0.0625, 0.125, 0.0625),
            y_axis: Vec3::new(0.125, 0.25, 0.125),
            z_axis: Vec3::new(0.0625, 0.125, 0.0625),
        },
    });
    uniform.buffer.write_buffer(&render_device, &render_queue);
    counter.0 += 1;
}
