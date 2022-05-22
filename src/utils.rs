use bevy::{
    core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d},
    prelude::*,
    render::{
        camera::{ActiveCamera, Camera3d, ExtractedCamera},
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotValue},
        render_phase::RenderPhase,
        renderer::RenderContext,
        view::{ExtractedView, VisibleEntities},
    },
};

use crate::simple_3d_graph;

/// A render node that executes simple graph for given camera of type `M`.
pub struct SimplePassDriver<M: Component + Default> {
    query: QueryState<Entity, With<M>>,
}

impl<M: Component + Default> SimplePassDriver<M> {
    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl<M: Component + Default> Node for SimplePassDriver<M> {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        for camera in self.query.iter_manual(world) {
            graph.run_sub_graph(simple_3d_graph::NAME, vec![SlotValue::Entity(camera)])?;
        }

        Ok(())
    }
}

/// Extract all cameras of type `M`, as [`extract_cameras`](bevy::render::camera::extract_cameras) only extracts active cameras.
pub fn extract_custom_cameras<M: Component + Default>(
    mut commands: Commands,
    windows: Res<Windows>,
    images: Res<Assets<Image>>,
    cameras: Query<(Entity, &Camera, &GlobalTransform, &VisibleEntities), With<M>>,
) {
    for (entity, camera, transform, visible_entities) in cameras.iter() {
        if let Some(size) = camera.target.get_physical_size(&windows, &images) {
            commands.get_or_spawn(entity).insert_bundle((
                ExtractedCamera {
                    target: camera.target.clone(),
                    physical_size: camera.target.get_physical_size(&windows, &images),
                },
                ExtractedView {
                    projection: camera.projection_matrix,
                    transform: *transform,
                    width: size.x,
                    height: size.y,
                    near: camera.near,
                    far: camera.far,
                },
                visible_entities.clone(),
                M::default(),
                RenderPhase::<Opaque3d>::default(),
                RenderPhase::<AlphaMask3d>::default(),
                RenderPhase::<Transparent3d>::default(),
            ));
        }
    }
}

/// Sync the custom camera's states with the main camera.
pub fn update_custom_camera<M: Default + Component>(
    main_active: Res<ActiveCamera<Camera3d>>,
    custom_active: Res<ActiveCamera<M>>,
    mut query: Query<(&mut Transform, &mut Camera)>,
) {
    if let Some((main_camera, custom_camera)) = main_active.get().zip(custom_active.get()) {
        let [(main_transform, main_camera), (mut custom_transform, mut custom_camera)] =
            query.many_mut([main_camera, custom_camera]);
        *custom_transform = *main_transform;
        custom_camera.projection_matrix = main_camera.projection_matrix;
        custom_camera.depth_calculation = main_camera.depth_calculation;
    }
}

pub fn extract_custom_camera_phases<M: Default + Component>(
    mut commands: Commands,
    active: Res<ActiveCamera<M>>,
) {
    if let Some(entity) = active.get() {
        commands.get_or_spawn(entity).insert_bundle((
            RenderPhase::<Opaque3d>::default(),
            RenderPhase::<AlphaMask3d>::default(),
            RenderPhase::<Transparent3d>::default(),
        ));
    }
}
