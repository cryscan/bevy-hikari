//! # bevy-hikari
//!
//! An implementation of Voxel Cone Tracing Global Illumination for [bevy].
//!

use bevy::{
    core_pipeline::MainPass3dNode,
    prelude::*,
    render::{
        render_graph::{
            Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType, SlotValue,
        },
        renderer::RenderContext,
        view::Layer,
        RenderApp,
    },
};
use volume::VolumePlugin;

mod volume;

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_MIPMAP_LEVEL_COUNT: usize = 8;
pub const VOXEL_COUNT: usize = 16777216;

pub const VOXEL_LAYER: Layer = 16;
pub const IRRADIANCE_LAYER: Layer = 17;
pub const DEFERRED_LAYER: Layer = 18;

pub mod node {
    pub const VOXEL_PASS_DRIVER: &[&str] = &[
        "voxel_0_pass_driver",
        "voxel_1_pass_driver",
        "voxel_2_pass_driver",
    ];
}

pub mod simple_3d_graph {
    pub const NAME: &str = "simple_3d";
    pub mod input {
        pub const VIEW_ENTITY: &str = "view_entity";
    }
    pub mod node {
        pub const MAIN_PASS: &str = "main_pass";
    }
}

pub struct GiPlugin;
impl Plugin for GiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GiConfig>().add_plugin(VolumePlugin);

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let pass_node_3d = MainPass3dNode::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();

        let mut simple_3d_graph = RenderGraph::default();
        simple_3d_graph.add_node(simple_3d_graph::node::MAIN_PASS, pass_node_3d);
        let input_node_id = simple_3d_graph.set_input(vec![SlotInfo::new(
            simple_3d_graph::input::VIEW_ENTITY,
            SlotType::Entity,
        )]);
        simple_3d_graph
            .add_slot_edge(
                input_node_id,
                simple_3d_graph::input::VIEW_ENTITY,
                simple_3d_graph::node::MAIN_PASS,
                MainPass3dNode::IN_VIEW,
            )
            .unwrap();
        graph.add_sub_graph(simple_3d_graph::NAME, simple_3d_graph);
    }
}

pub struct GiConfig {
    pub global: bool,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self { global: true }
    }
}

/// Marker component for meshes not casting GI.
#[derive(Component)]
pub struct NotGiCaster;

/// Marker component for meshes not receiving GI.
#[derive(Component)]
pub struct NotGiReceiver;

/// A render node that executes `simple_3d_graph` for given camera of type `T`.
pub struct SimplePassDriver<T: Component + Default> {
    query: QueryState<Entity, With<T>>,
}
impl<T: Component + Default> SimplePassDriver<T> {
    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}
impl<T: Component + Default> Node for SimplePassDriver<T> {
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
