use crate::transform::GlobalTransformQueue;
use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
        RenderApp, RenderSet,
    },
};

pub struct ViewPlugin;
impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<FrameCounter>()
            .add_plugin(ExtractComponentPlugin::<FrameCounter>::default())
            .add_plugin(ExtractComponentPlugin::<FrameUniform>::default())
            .add_plugin(UniformComponentPlugin::<FrameUniform>::default())
            .add_system(frame_counter_system.in_base_set(CoreSet::PostUpdate));

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<PreviousViewUniforms>()
                .add_system(prepare_view_uniforms.in_set(RenderSet::Prepare));
        }
    }
}

#[derive(Clone, ShaderType)]
pub struct PreviousViewUniform {
    view_proj: Mat4,
    inverse_view_proj: Mat4,
}

#[derive(Default, Resource)]
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

#[derive(Default, Clone, Copy, Component, Reflect, Deref, DerefMut, ExtractComponent)]
#[reflect(Component)]
pub struct FrameCounter(pub usize);

#[allow(clippy::type_complexity)]
fn frame_counter_system(
    mut commands: Commands,
    mut queries: ParamSet<(
        Query<Entity, (With<Camera>, Without<FrameCounter>)>,
        Query<&mut FrameCounter>,
    )>,
) {
    for entity in &queries.p0() {
        commands.entity(entity).insert(FrameCounter::default());
    }

    for mut counter in queries.p1().iter_mut() {
        **counter += 1;
    }
}

#[derive(Debug, Default, Clone, Copy, Component, ShaderType, ExtractComponent)]
pub struct FrameUniform {
    pub kernel: Mat3,
    pub halton: [Vec4; 8],
    pub clear_color: Vec4,
    pub number: u32,
    pub direct_validate_interval: u32,
    pub emissive_validate_interval: u32,
    pub indirect_bounces: u32,
    pub temporal_reuse: u32,
    pub emissive_spatial_reuse: u32,
    pub indirect_spatial_reuse: u32,
    pub max_temporal_reuse_count: u32,
    pub max_spatial_reuse_count: u32,
    pub max_reservoir_lifetime: f32,
    pub solar_angle: f32,
    pub max_indirect_luminance: f32,
    pub upscale_ratio: f32,
}

const KERNEL: Mat3 = Mat3 {
    x_axis: Vec3::new(0.0625, 0.125, 0.0625),
    y_axis: Vec3::new(0.125, 0.25, 0.125),
    z_axis: Vec3::new(0.0625, 0.125, 0.0625),
};
const HALTON: [Vec4; 8] = [
    Vec4::new(0.000000, 0.000000, 0.500000, 0.333333),
    Vec4::new(0.250000, 0.666667, 0.750000, 0.111111),
    Vec4::new(0.125000, 0.444444, 0.625000, 0.777778),
    Vec4::new(0.375000, 0.222222, 0.875000, 0.555556),
    Vec4::new(0.062500, 0.888889, 0.562500, 0.037037),
    Vec4::new(0.312500, 0.370370, 0.812500, 0.703704),
    Vec4::new(0.187500, 0.148148, 0.687500, 0.481481),
    Vec4::new(0.437500, 0.814815, 0.937500, 0.259259),
];
