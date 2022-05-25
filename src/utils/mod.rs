use crate::simple_3d_graph;
use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotValue},
        renderer::RenderContext,
    },
};

pub mod custom_camera;

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
