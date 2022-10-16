use bevy::{
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        view::VisibilitySystems,
    },
    transform::TransformSystem,
};

pub struct TransformPlugin;
impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<GlobalTransformQueue>::default())
            .add_system_to_stage(
                CoreStage::PostUpdate,
                previous_transform_system.after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                hierarchy_visibility_system.after(VisibilitySystems::CheckVisibility),
            );
    }
}

#[derive(Component, Debug, Clone, Deref, DerefMut)]
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
    mut queries: ParamSet<(
        Query<(Entity, &GlobalTransform), Without<GlobalTransformQueue>>,
        Query<(&GlobalTransform, &mut GlobalTransformQueue)>,
    )>,
) {
    for (entity, transform) in &queries.p0() {
        let matrix = transform.compute_matrix();
        commands
            .entity(entity)
            .insert(GlobalTransformQueue([matrix; 2]));
    }

    for (transform, mut queue) in &mut queries.p1() {
        queue[1] = queue[0];
        queue[0] = transform.compute_matrix();
    }
}

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct HierarchyVisibility(pub bool);

fn hierarchy_visibility_system(
    mut commands: Commands,
    mut queries: ParamSet<(
        Query<(Entity, &ComputedVisibility), Without<HierarchyVisibility>>,
        Query<(&ComputedVisibility, &mut HierarchyVisibility)>,
    )>,
) {
    for (entity, computed) in &queries.p0() {
        commands
            .entity(entity)
            .insert(HierarchyVisibility(computed.is_visible_in_hierarchy()));
    }

    for (computed, mut hierarchy) in &mut queries.p1() {
        let visible = computed.is_visible_in_hierarchy();
        if visible != hierarchy.0 {
            hierarchy.0 = visible;
        }
    }
}
