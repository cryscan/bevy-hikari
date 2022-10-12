use crate::{
    light::{LightPassNode, LightPlugin},
    mesh_material::MeshMaterialPlugin,
    overlay::{OverlayPassNode, OverlayPlugin},
    prepass::{PrepassNode, PrepassPlugin},
    transform::TransformPlugin,
    view::ViewPlugin,
};
use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::MainPass3dNode,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_graph::{RenderGraph, SlotInfo, SlotType},
        texture::{CompressedImageFormats, ImageType},
        Extract, RenderApp, RenderStage,
    },
};
use std::f32::consts::PI;

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

pub const UTILS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4462033275253590181);
pub const MESH_VIEW_TYPES_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 10086770709483722043);
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

pub struct HikariPlugin;
impl Plugin for HikariPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            UTILS_SHADER_HANDLE,
            "shaders/utils.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_VIEW_TYPES_HANDLE,
            "shaders/mesh_view_types.wgsl",
            Shader::from_wgsl
        );
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

        let noise_load_system = move |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
            let bytes = [
                include_bytes!("noise/LDR_RGBA_0.png"),
                include_bytes!("noise/LDR_RGBA_1.png"),
                include_bytes!("noise/LDR_RGBA_2.png"),
                include_bytes!("noise/LDR_RGBA_3.png"),
                include_bytes!("noise/LDR_RGBA_4.png"),
                include_bytes!("noise/LDR_RGBA_5.png"),
                include_bytes!("noise/LDR_RGBA_6.png"),
                include_bytes!("noise/LDR_RGBA_7.png"),
                include_bytes!("noise/LDR_RGBA_8.png"),
                include_bytes!("noise/LDR_RGBA_9.png"),
                include_bytes!("noise/LDR_RGBA_10.png"),
                include_bytes!("noise/LDR_RGBA_11.png"),
                include_bytes!("noise/LDR_RGBA_12.png"),
                include_bytes!("noise/LDR_RGBA_13.png"),
                include_bytes!("noise/LDR_RGBA_14.png"),
                include_bytes!("noise/LDR_RGBA_15.png"),
                include_bytes!("noise/LDR_RGBA_16.png"),
                include_bytes!("noise/LDR_RGBA_17.png"),
                include_bytes!("noise/LDR_RGBA_18.png"),
                include_bytes!("noise/LDR_RGBA_19.png"),
                include_bytes!("noise/LDR_RGBA_20.png"),
                include_bytes!("noise/LDR_RGBA_21.png"),
                include_bytes!("noise/LDR_RGBA_22.png"),
                include_bytes!("noise/LDR_RGBA_23.png"),
                include_bytes!("noise/LDR_RGBA_24.png"),
                include_bytes!("noise/LDR_RGBA_25.png"),
                include_bytes!("noise/LDR_RGBA_26.png"),
                include_bytes!("noise/LDR_RGBA_27.png"),
                include_bytes!("noise/LDR_RGBA_28.png"),
                include_bytes!("noise/LDR_RGBA_29.png"),
                include_bytes!("noise/LDR_RGBA_30.png"),
                include_bytes!("noise/LDR_RGBA_31.png"),
                include_bytes!("noise/LDR_RGBA_32.png"),
                include_bytes!("noise/LDR_RGBA_33.png"),
                include_bytes!("noise/LDR_RGBA_34.png"),
                include_bytes!("noise/LDR_RGBA_35.png"),
                include_bytes!("noise/LDR_RGBA_36.png"),
                include_bytes!("noise/LDR_RGBA_37.png"),
                include_bytes!("noise/LDR_RGBA_38.png"),
                include_bytes!("noise/LDR_RGBA_39.png"),
                include_bytes!("noise/LDR_RGBA_40.png"),
                include_bytes!("noise/LDR_RGBA_41.png"),
                include_bytes!("noise/LDR_RGBA_42.png"),
                include_bytes!("noise/LDR_RGBA_43.png"),
                include_bytes!("noise/LDR_RGBA_44.png"),
                include_bytes!("noise/LDR_RGBA_45.png"),
                include_bytes!("noise/LDR_RGBA_46.png"),
                include_bytes!("noise/LDR_RGBA_47.png"),
                include_bytes!("noise/LDR_RGBA_48.png"),
                include_bytes!("noise/LDR_RGBA_49.png"),
                include_bytes!("noise/LDR_RGBA_50.png"),
                include_bytes!("noise/LDR_RGBA_51.png"),
                include_bytes!("noise/LDR_RGBA_52.png"),
                include_bytes!("noise/LDR_RGBA_53.png"),
                include_bytes!("noise/LDR_RGBA_54.png"),
                include_bytes!("noise/LDR_RGBA_55.png"),
                include_bytes!("noise/LDR_RGBA_56.png"),
                include_bytes!("noise/LDR_RGBA_57.png"),
                include_bytes!("noise/LDR_RGBA_58.png"),
                include_bytes!("noise/LDR_RGBA_59.png"),
                include_bytes!("noise/LDR_RGBA_60.png"),
                include_bytes!("noise/LDR_RGBA_61.png"),
                include_bytes!("noise/LDR_RGBA_62.png"),
                include_bytes!("noise/LDR_RGBA_63.png"),
            ];
            let handles = Vec::from(bytes.map(|buffer| {
                let image = Image::from_buffer(
                    buffer,
                    ImageType::Extension("png"),
                    CompressedImageFormats::NONE,
                    false,
                )
                .unwrap();
                images.add(image)
            }));

            commands.insert_resource(NoiseTexture(handles));
        };

        app.init_resource::<HikariConfig>()
            .add_plugin(TransformPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshMaterialPlugin)
            .add_plugin(PrepassPlugin)
            .add_plugin(LightPlugin)
            .add_plugin(OverlayPlugin)
            .add_startup_system(noise_load_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_system_to_stage(RenderStage::Extract, extract_config);

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

#[derive(Clone)]
pub struct HikariConfig {
    /// The interval of frames between sample validation passes.
    pub validation_interval: usize,
    /// Chance for the indirect rays to bounce again after first hit.
    pub second_bounce_chance: f32,
    /// Half angle of the solar cone apex in radians.
    pub solar_angle: f32,
}

impl Default for HikariConfig {
    fn default() -> Self {
        Self {
            validation_interval: 4,
            second_bounce_chance: 0.25,
            solar_angle: PI / 36.0,
        }
    }
}

fn extract_config(mut commands: Commands, config: Extract<Res<HikariConfig>>) {
    commands.insert_resource(config.clone());
}

#[derive(Clone, Deref, DerefMut)]
pub struct NoiseTexture(pub Vec<Handle<Image>>);
