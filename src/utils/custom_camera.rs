use bevy::{
    core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d},
    pbr::*,
    prelude::*,
    render::{
        camera::{ActiveCamera, Camera3d, ExtractedCamera},
        render_phase::RenderPhase,
        view::{ExtractedView, VisibleEntities},
    },
};

/// Sync the custom camera's states with the main camera.
pub fn update_transform<M: Default + Component>(
    main_active: Res<ActiveCamera<Camera3d>>,
    custom_active: Res<ActiveCamera<M>>,
    mut query: Query<&mut Transform>,
) {
    if let Some((main, custom)) = main_active.get().zip(custom_active.get()) {
        let [main_transform, mut custom_transform] = query.many_mut([main, custom]);
        *custom_transform = *main_transform;
    }
}

/// Extract all cameras of type `M`, as [`extract_cameras`](bevy::render::camera::extract_cameras) only extracts active cameras.
pub fn extract_camera<M: Component + Default>(
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
                // Prevent lights from being prepared automatically.
                // RenderPhase::<Opaque3d>::default(),
                // RenderPhase::<AlphaMask3d>::default(),
                // RenderPhase::<Transparent3d>::default(),
            ));
        }
    }
}

pub fn extract_phases<M: Default + Component>(
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

/// Hijack main camera's [`ViewShadowBindings`](bevy::pbr::ViewShadowBindings).
pub fn prepare_lights<M: Default + Component>(
    mut commands: Commands,
    active: Res<ActiveCamera<Camera3d>>,
    query: Query<
        (
            &ViewShadowBindings,
            &ViewLightEntities,
            &ViewLightsUniformOffset,
        ),
        Without<M>,
    >,
    cameras: Query<Entity, With<M>>,
) {
    if let Some(main_camera) = active.get() {
        if let Ok((view_shadow_bindings, view_light_entities, view_lights_uniform_offset)) =
            query.get(main_camera)
        {
            for entity in cameras.iter() {
                let ViewShadowBindings {
                    point_light_depth_texture,
                    point_light_depth_texture_view,
                    directional_light_depth_texture,
                    directional_light_depth_texture_view,
                } = view_shadow_bindings;
                let (
                    point_light_depth_texture,
                    point_light_depth_texture_view,
                    directional_light_depth_texture,
                    directional_light_depth_texture_view,
                ) = (
                    point_light_depth_texture.clone(),
                    point_light_depth_texture_view.clone(),
                    directional_light_depth_texture.clone(),
                    directional_light_depth_texture_view.clone(),
                );

                commands.entity(entity).insert_bundle((
                    ViewShadowBindings {
                        point_light_depth_texture,
                        point_light_depth_texture_view,
                        directional_light_depth_texture,
                        directional_light_depth_texture_view,
                    },
                    ViewLightEntities {
                        lights: view_light_entities.lights.clone(),
                    },
                    ViewLightsUniformOffset {
                        offset: view_lights_uniform_offset.offset,
                    },
                    RenderPhase::<Opaque3d>::default(),
                    RenderPhase::<AlphaMask3d>::default(),
                    RenderPhase::<Transparent3d>::default(),
                ));
            }
        }
    }
}
