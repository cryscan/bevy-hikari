//! # bevy-hikari
//!
//! An implementation of Voxel Cone Tracing Global Illumination for [bevy].
//!

use bevy::{
    core_pipeline::MainPass3dNode,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_graph::{RenderGraph, SlotInfo, SlotType},
        RenderApp,
    },
};
use mipmap::MipmapPlugin;
use volume::VolumePlugin;

mod deferred;
mod mipmap;
mod utils;
mod volume;

pub const VOLUME_STRUCT_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 16383356904282015386);
pub const STANDARD_MATERIAL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5199983296924258000);
pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984740);
pub const MIPMAP_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5437952701024607848);
pub const ALBEDO_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 1569823627703093993);
pub const IRRADIANCE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14497829665752154997);
pub const OVERLAY_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 15849919474767323744);

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_MIPMAP_LEVEL_COUNT: usize = 8;
pub const VOXEL_COUNT: usize = 16777216;

pub mod node {
    pub const VOXEL_PASS_DRIVER: &str = "voxel_pass_driver";
    pub const VOXEL_CLEAR_PASS: &str = "voxel_clear_pass";
    pub const MIPMAP_PASS: &str = "mipmap_pass";
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
        app.add_plugin(VolumePlugin).add_plugin(MipmapPlugin);

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOLUME_STRUCT_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/volume_struct.wgsl")),
        );
        shaders.set_untracked(
            STANDARD_MATERIAL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/standard_material.wgsl")),
        );
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl")),
        );
        shaders.set_untracked(
            MIPMAP_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/mipmap.wgsl")),
        );
        shaders.set_untracked(
            ALBEDO_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/albedo.wgsl")),
        );

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

/// Marker component for meshes not casting GI.
#[derive(Component)]
pub struct NotGiCaster;

/// Marker component for meshes not receiving GI.
#[derive(Component)]
pub struct NotGiReceiver;
