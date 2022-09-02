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
            previous_transform_system.before(TransformSystem::TransformPropagate),
        );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_system_to_stage(RenderStage::Extract, extract_previous_transform);
        }
    }
}

#[derive(Component, Debug, PartialEq, Clone, Copy, Deref, DerefMut)]
pub struct PreviousGlobalTransform(GlobalTransform);

fn previous_transform_system(
    mut commands: Commands,
    setup_query: Query<(Entity, &GlobalTransform), Without<PreviousGlobalTransform>>,
    mut update_query: Query<(&GlobalTransform, &mut PreviousGlobalTransform)>,
) {
    for (entity, transform) in &setup_query {
        commands
            .entity(entity)
            .insert(PreviousGlobalTransform(*transform));
    }

    for (transform, mut previous_transform) in &mut update_query {
        previous_transform.0 = *transform;
    }
}

fn extract_previous_transform(
    mut commands: Commands,
    query: Extract<Query<(Entity, &PreviousGlobalTransform)>>,
) {
    for (entity, transform) in query.iter() {
        commands.get_or_spawn(entity).insert(*transform);
    }
}
