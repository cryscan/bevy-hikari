//! # bevy-hikari
//!
//! An implementation of Voxel Cone Tracing Global Illumination for [bevy].
//!

use bevy::{
    core_pipeline::{AlphaMask3d, MainPass3dNode, Opaque3d, Transparent3d},
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::ExtractedCamera,
        render_graph::{
            Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType, SlotValue,
        },
        render_phase::RenderPhase,
        renderer::RenderContext,
        view::{ExtractedView, RenderLayers, VisibleEntities},
        RenderApp,
    },
};
use mipmap::MipmapPlugin;
use volume::VolumePlugin;

mod mipmap;
mod volume;

pub const VOLUME_STRUCT_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 16383356904282015386);
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
        app.init_resource::<GiConfig>()
            .add_plugin(VolumePlugin)
            .add_plugin(MipmapPlugin);

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOLUME_STRUCT_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/volume_struct.wgsl")),
        );
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl")),
        );
        shaders.set_untracked(
            MIPMAP_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/mipmap.wgsl")),
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

pub struct GiConfig {
    pub global: bool,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self { global: true }
    }
}

pub enum GiRenderLayers {
    Voxel = 16,
    Deferred = 17,
    Irradiance = 18,
    Overlay = 19,
}

impl From<GiRenderLayers> for RenderLayers {
    fn from(layer: GiRenderLayers) -> Self {
        Self::layer(layer as u8)
    }
}

/// Marker component for meshes not casting GI.
#[derive(Component)]
pub struct NotGiCaster;

/// Marker component for meshes not receiving GI.
#[derive(Component)]
pub struct NotGiReceiver;

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

/// Manually extract all cameras of type `M`, as [`CameraTypePlugin`](bevy::render::camera::CameraTypePlugin) only extracts the active camera.
pub fn extract_custom_cameras<M: Component + Default>(
    mut commands: Commands,
    windows: Res<Windows>,
    images: Res<Assets<Image>>,
    cameras: Query<(Entity, &Camera, &GlobalTransform, &VisibleEntities), With<M>>,
) {
    for (entity, camera, transform, visible_entities) in cameras.iter() {
        if let Some(size) = camera.target.get_physical_size(&windows, &images) {
            commands.get_or_spawn(entity).insert_bundle((
                ExtractedCamera {
                    target: camera.target.clone(),
                    physical_size: camera.target.get_physical_size(&windows, &images),
                },
                ExtractedView {
                    projection: camera.projection_matrix,
                    transform: *transform,
                    width: size.x,
                    height: size.y,
                    near: camera.near,
                    far: camera.far,
                },
                visible_entities.clone(),
                M::default(),
                RenderPhase::<Opaque3d>::default(),
                RenderPhase::<AlphaMask3d>::default(),
                RenderPhase::<Transparent3d>::default(),
            ));
        }
    }
}
