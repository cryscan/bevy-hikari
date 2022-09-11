use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::MainPass3dNode,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_graph::{RenderGraph, SlotInfo, SlotType},
        RenderApp,
    },
};
use illumination::LightPlugin;
use mesh::MeshPlugin;
use prepass::{PrepassNode, PrepassPlugin};
use transform::TransformPlugin;
use view::ViewPlugin;

use crate::illumination::LightPassNode;

pub mod illumination;
pub mod mesh;
pub mod prelude;
pub mod prepass;
pub mod transform;
pub mod view;

pub mod graph {
    pub const NAME: &str = "hikari";
    pub mod input {
        pub const VIEW_ENTITY: &str = "view_entity";
    }
    pub mod node {
        pub const PREPASS: &str = "prepass";
        pub const LIGHT_DIRECT_PASS: &str = "light_direct_pass";
        pub const LIGHT_INDIRECT_PASS: &str = "light_indirect_pass";
    }
}

pub const PREPASS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4693612430004931427);
pub const LIGHT_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 9657319286592943583);
pub const RAY_TRACING_TYPES_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 15819591594687298858);
pub const RAY_TRACING_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5025976374517268);
// pub const RAY_TRACING_FUNCTIONS_HANDLE: HandleUntyped =
//     HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 6789977396118176997);

pub struct HikariPlugin;
impl Plugin for HikariPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PREPASS_SHADER_HANDLE,
            "shaders/prepass.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            LIGHT_SHADER_HANDLE,
            "shaders/illumination.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RAY_TRACING_TYPES_HANDLE,
            "shaders/ray_tracing_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RAY_TRACING_BINDINGS_HANDLE,
            "shaders/ray_tracing_bindings.wgsl",
            Shader::from_wgsl
        );
        // load_internal_asset!(
        //     app,
        //     RAY_TRACING_FUNCTIONS_HANDLE,
        //     "shaders/ray_tracing_functions.wgsl",
        //     Shader::from_wgsl
        // );

        app.add_plugin(TransformPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshPlugin)
            .add_plugin(PrepassPlugin)
            .add_plugin(LightPlugin);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let prepass_node = PrepassNode::new(&mut render_app.world);
            let light_pass_node = LightPassNode::new(&mut render_app.world);
            let pass_node_3d = MainPass3dNode::new(&mut render_app.world);

            let mut graph = render_app.world.resource_mut::<RenderGraph>();

            let mut hikari_graph = RenderGraph::default();
            hikari_graph.add_node(graph::node::PREPASS, prepass_node);
            hikari_graph.add_node(graph::node::LIGHT_DIRECT_PASS, light_pass_node);
            hikari_graph.add_node(
                bevy::core_pipeline::core_3d::graph::node::MAIN_PASS,
                pass_node_3d,
            );
            let input_node_id = hikari_graph.set_input(vec![SlotInfo::new(
                graph::input::VIEW_ENTITY,
                SlotType::Entity,
            )]);
            hikari_graph
                .add_slot_edge(
                    input_node_id,
                    graph::input::VIEW_ENTITY,
                    graph::node::PREPASS,
                    PrepassNode::IN_VIEW,
                )
                .unwrap();
            hikari_graph
                .add_slot_edge(
                    input_node_id,
                    graph::input::VIEW_ENTITY,
                    graph::node::LIGHT_DIRECT_PASS,
                    LightPassNode::IN_VIEW,
                )
                .unwrap();
            hikari_graph
                .add_node_edge(graph::node::PREPASS, graph::node::LIGHT_DIRECT_PASS)
                .unwrap();
            hikari_graph
                .add_slot_edge(
                    input_node_id,
                    graph::input::VIEW_ENTITY,
                    bevy::core_pipeline::core_3d::graph::node::MAIN_PASS,
                    MainPass3dNode::IN_VIEW,
                )
                .unwrap();
            graph.add_sub_graph(graph::NAME, hikari_graph);
        }
    }
}
