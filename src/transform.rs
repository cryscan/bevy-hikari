use bevy::{
    prelude::*,
    render::{Extract, RenderApp, RenderStage},
    transform::TransformSystem,
};

pub struct TransformPlugin;
impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(
            CoreStage::PostUpdate,
            previous_transform_system.after(TransformSystem::TransformPropagate),
        );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_system_to_stage(RenderStage::Extract, extract_previous_transform);
        }
    }
}

#[derive(Component, Debug, PartialEq, Clone, Copy, Deref, DerefMut)]
pub struct GlobalTransformQueue(pub [Mat4; 2]);

fn previous_transform_system(
    mut commands: Commands,
    setup_query: Query<(Entity, &GlobalTransform), Without<GlobalTransformQueue>>,
    mut update_query: Query<(&GlobalTransform, &mut GlobalTransformQueue)>,
) {
    for (entity, transform) in &setup_query {
        let matrix = transform.compute_matrix();
        commands
            .entity(entity)
            .insert(GlobalTransformQueue([matrix; 2]));
    }

    for (transform, mut queue) in &mut update_query {
        queue[1] = queue[0];
        queue[0] = transform.compute_matrix();
    }
}

fn extract_previous_transform(
    mut commands: Commands,
    query: Extract<Query<(Entity, &GlobalTransformQueue)>>,
) {
    for (entity, queue) in query.iter() {
        commands.get_or_spawn(entity).insert(*queue);
    }
}
