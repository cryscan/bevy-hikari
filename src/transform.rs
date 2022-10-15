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

#[derive(Component, Debug, PartialEq, Clone, Copy, Deref, DerefMut)]
pub struct GlobalTransformQueue(pub [Mat4; 2]);

impl ExtractComponent for GlobalTransformQueue {
    type Query = &'static Self;
    type Filter = ();

    fn extract_component(item: QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

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
