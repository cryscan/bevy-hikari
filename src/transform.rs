use bevy::{
    ecs::query::QueryItem,
    prelude::*,
    render::extract_component::{ExtractComponent, ExtractComponentPlugin},
    transform::TransformSystem,
};

pub struct TransformPlugin;
impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<GlobalTransformQueue>::default())
            .add_system_to_stage(
                CoreStage::PostUpdate,
                previous_transform_system.after(TransformSystem::TransformPropagate),
            );
    }
}

#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GlobalTransformQueue(pub [Mat4; 2]);

impl ExtractComponent for GlobalTransformQueue {
    type Query = &'static Self;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<Self::Query>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

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
