use bevy::{
    prelude::*,
    render::extract_component::{ExtractComponent, ExtractComponentPlugin},
    transform::TransformSystem,
};

pub struct TransformPlugin;
impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<GlobalTransformQueue>::default())
            .add_system(
                previous_transform_system
                    .in_base_set(CoreSet::PostUpdate)
                    .after(TransformSystem::TransformPropagate),
            );
    }
}

#[derive(Component, Debug, Clone, Deref, DerefMut, ExtractComponent)]
pub struct GlobalTransformQueue(pub [Mat4; 2]);

#[allow(clippy::type_complexity)]
fn previous_transform_system(
    mut commands: Commands,
    query: Query<(Entity, &GlobalTransform, Option<&GlobalTransformQueue>)>,
) {
    for (entity, transform, queue) in &query {
        let matrix = transform.compute_matrix();
        let queue = match queue {
            Some(queue) => GlobalTransformQueue([matrix, queue[0]]),
            None => GlobalTransformQueue([matrix; 2]),
        };
        commands.entity(entity).insert(queue);
    }
}
