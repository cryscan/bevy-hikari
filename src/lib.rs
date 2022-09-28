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
use light::{LightPassNode, LightPlugin};
use mesh_material::MeshMaterialPlugin;
use overlay::{OverlayPassNode, OverlayPlugin};
use prepass::{PrepassNode, PrepassPlugin};
use transform::TransformPlugin;
use view::ViewPlugin;

pub mod light;
pub mod mesh_material;
pub mod overlay;
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
        pub const OVERLAY_PASS: &str = "overlay_pass";
    }
}

pub const WORKGROUP_SIZE: u32 = 8;
pub const NOISE_TEXTURE_COUNT: usize = 64;

pub const MESH_VIEW_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 8835349515886344623);
pub const MESH_MATERIAL_TYPES_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 15819591594687298858);
pub const MESH_MATERIAL_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5025976374517268);
pub const DEFERRED_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14467895678105108252);
pub const PREPASS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4693612430004931427);
pub const LIGHT_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 9657319286592943583);
pub const OVERLAY_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 10969344919103020615);
pub const QUAD_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Mesh::TYPE_UUID, 4740146776519512271);

pub struct HikariPlugin {
    noise_folder: String,
}

impl HikariPlugin {
    pub fn new(noise_folder: &str) -> Self {
        Self {
            noise_folder: noise_folder.into(),
        }
    }
}

impl Default for HikariPlugin {
    fn default() -> Self {
        Self {
            noise_folder: "textures/blue_noise".into(),
        }
    }
}

#[derive(Clone, Deref, DerefMut)]
pub struct NoiseTexture(pub Vec<Handle<Image>>);

impl Plugin for HikariPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            MESH_VIEW_BINDINGS_HANDLE,
            "shaders/mesh_view_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_MATERIAL_TYPES_HANDLE,
            "shaders/mesh_material_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_MATERIAL_BINDINGS_HANDLE,
            "shaders/mesh_material_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DEFERRED_BINDINGS_HANDLE,
            "shaders/deferred_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PREPASS_SHADER_HANDLE,
            "shaders/prepass.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            LIGHT_SHADER_HANDLE,
            "shaders/light.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            OVERLAY_SHADER_HANDLE,
            "shaders/overlay.wgsl",
            Shader::from_wgsl
        );

        let noise_path = self.noise_folder.clone();
        let load_system = move |mut commands: Commands, asset_server: Res<AssetServer>| {
            let handles = (0..NOISE_TEXTURE_COUNT)
                .map(|id| {
                    let name = format!("{}/LDR_RGBA_{}.png", noise_path, id);
                    asset_server.load(&name)
                })
                .collect();
            commands.insert_resource(NoiseTexture(handles));
        };

        app.add_plugin(TransformPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshMaterialPlugin)
            .add_plugin(PrepassPlugin)
            .add_plugin(LightPlugin)
            .add_plugin(OverlayPlugin)
            .add_startup_system(load_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let prepass_node = PrepassNode::new(&mut render_app.world);
            let light_pass_node = LightPassNode::new(&mut render_app.world);
            let overlay_pass_node = OverlayPassNode::new(&mut render_app.world);
            // let pass_node_3d = MainPass3dNode::new(&mut render_app.world);

            let mut graph = render_app.world.resource_mut::<RenderGraph>();

            let mut hikari_graph = RenderGraph::default();
            hikari_graph.add_node(graph::node::PREPASS, prepass_node);
            hikari_graph.add_node(graph::node::LIGHT_DIRECT_PASS, light_pass_node);
            hikari_graph.add_node(graph::node::OVERLAY_PASS, overlay_pass_node);
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
                    graph::node::OVERLAY_PASS,
                    MainPass3dNode::IN_VIEW,
                )
                .unwrap();
            hikari_graph
                .add_node_edge(graph::node::LIGHT_DIRECT_PASS, graph::node::OVERLAY_PASS)
                .unwrap();
            graph.add_sub_graph(graph::NAME, hikari_graph);
        }
    }
}
