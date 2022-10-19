use crate::{transform::GlobalTransformQueue, HikariConfig};
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
                .init_resource::<FrameUniformBuffer>()
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
    views: Query<(Entity, &ExtractedView, &GlobalTransformQueue)>,
) {
    view_uniforms.uniforms.clear();
    for (entity, camera, queue) in &views {
        let projection = camera.projection;
        let inverse_projection = projection.inverse();
        let view = queue[1];
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
pub struct FrameUniform {
    pub kernel: Mat3,
    pub clear_color: Vec4,
    pub number: u32,
    pub validation_interval: u32,
    pub max_temporal_reuse_count: u32,
    pub max_spatial_reuse_count: u32,
    pub direct_oversample_threshold: u32,
    pub solar_angle: f32,
    pub max_indirect_luminance: f32,
    pub suppress_temporal_accum: u32,
}

#[derive(Default)]
pub struct FrameUniformBuffer {
    pub buffer: UniformBuffer<FrameUniform>,
}

fn prepare_frame_uniform(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    config: Res<HikariConfig>,
    clear_color: Res<ClearColor>,
    mut uniform: ResMut<FrameUniformBuffer>,
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

    let HikariConfig {
        validation_interval,
        max_temporal_reuse_count,
        max_spatial_reuse_count,
        direct_oversample_threshold,
        solar_angle,
        max_indirect_luminance,
        suppress_temporal_accum,
        ..
    } = config.into_inner().clone();

    let validation_interval = validation_interval as u32;
    let max_temporal_reuse_count = max_temporal_reuse_count as u32;
    let max_spatial_reuse_count = max_spatial_reuse_count as u32;
    let direct_oversample_threshold = direct_oversample_threshold as u32;
    let suppress_temporal_accum = if suppress_temporal_accum { 1u32 } else { 0u32 };

    uniform.buffer.set(FrameUniform {
        kernel: Mat3 {
            x_axis: Vec3::new(0.0625, 0.125, 0.0625),
            y_axis: Vec3::new(0.125, 0.25, 0.125),
            z_axis: Vec3::new(0.0625, 0.125, 0.0625),
        },
        clear_color: clear_color.0.into(),
        number: counter.0 as u32,
        validation_interval,
        max_temporal_reuse_count,
        max_spatial_reuse_count,
        direct_oversample_threshold,
        solar_angle,
        max_indirect_luminance,
        suppress_temporal_accum,
    });
    uniform.buffer.write_buffer(&render_device, &render_queue);
    counter.0 += 1;
}
