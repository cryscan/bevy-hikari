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
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{RenderGraph, SlotInfo, SlotType},
        render_resource::*,
        renderer::RenderDevice,
        texture::{CompressedImageFormats, FallbackImage, ImageType},
        RenderApp,
    },
};
use std::{f32::consts::PI, num::NonZeroU32};

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
        pub const LIGHT_PASS: &str = "light_pass";
        pub const OVERLAY_PASS: &str = "overlay_pass";
    }
}

pub const WORKGROUP_SIZE: u32 = 8;
pub const NOISE_TEXTURE_COUNT: usize = 16;

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
            commands.insert_resource(NoiseTextures(handles));
        };

        app.register_type::<HikariConfig>()
            .init_resource::<HikariConfig>()
            .add_plugin(ExtractResourcePlugin::<HikariConfig>::default())
            .add_plugin(ExtractResourcePlugin::<NoiseTextures>::default())
            .add_plugin(TransformPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshMaterialPlugin)
            .add_plugin(PrepassPlugin)
            .add_plugin(LightPlugin)
            .add_plugin(OverlayPlugin)
            .add_startup_system(noise_load_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let prepass_node = PrepassNode::new(&mut render_app.world);
            let light_pass_node = LightPassNode::new(&mut render_app.world);
            let overlay_pass_node = OverlayPassNode::new(&mut render_app.world);
            // let pass_node_3d = MainPass3dNode::new(&mut render_app.world);

            let mut graph = render_app.world.resource_mut::<RenderGraph>();

            let mut hikari_graph = RenderGraph::default();
            hikari_graph.add_node(graph::node::PREPASS, prepass_node);
            hikari_graph.add_node(graph::node::LIGHT_PASS, light_pass_node);
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
                    graph::node::LIGHT_PASS,
                    LightPassNode::IN_VIEW,
                )
                .unwrap();
            hikari_graph
                .add_node_edge(graph::node::PREPASS, graph::node::LIGHT_PASS)
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
                .add_node_edge(graph::node::LIGHT_PASS, graph::node::OVERLAY_PASS)
                .unwrap();
            graph.add_sub_graph(graph::NAME, hikari_graph);
        }
    }
}

#[derive(Debug, Clone, ExtractResource, Reflect)]
#[reflect(Resource)]
pub struct HikariConfig {
    /// The interval of frames between sample validation passes.
    pub validation_interval: usize,
    /// Temporal reservoir sample count is capped by this value.
    pub max_temporal_reuse_count: usize,
    /// Spatial reservoir sample count is capped by this value.
    pub max_spatial_reuse_count: usize,
    /// Threshold for oversampling the direct illumination if the sample count is low.
    pub direct_oversample_threshold: usize,
    /// Half angle of the solar cone apex in radians.
    pub solar_angle: f32,
    /// Threshold that emissive objects begin to lit others.
    pub emissive_threshold: f32,
    /// Threshold for the indirect luminance to reduce fireflies.
    pub max_indirect_luminance: f32,
    /// Whether to do temporal sample reuse in ReSTIR.
    pub temporal_reuse: bool,
    /// Whether to do spatial sample reuse in ReSTIR.
    pub spatial_reuse: bool,
    /// Whether to perform spatial denoise for direct illumination.
    pub direct_spatial_denoise: bool,
    /// Whether to perform spatial denoise for indirect illumination.
    pub indirect_spatial_denoise: bool,
    /// Whether to perform TAA.
    pub temporal_anti_aliasing: bool,
}

impl Default for HikariConfig {
    fn default() -> Self {
        Self {
            validation_interval: 4,
            max_temporal_reuse_count: 50,
            max_spatial_reuse_count: 500,
            direct_oversample_threshold: 1,
            solar_angle: PI / 36.0,
            emissive_threshold: 0.00390625,
            max_indirect_luminance: 10.0,
            temporal_reuse: true,
            spatial_reuse: false,
            direct_spatial_denoise: true,
            indirect_spatial_denoise: true,
            temporal_anti_aliasing: true,
        }
    }
}

#[derive(Clone, Deref, ExtractResource)]
pub struct NoiseTextures(pub Vec<Handle<Image>>);

impl AsBindGroup for NoiseTextures {
    type Data = ();

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        images: &RenderAssets<Image>,
        fallback_image: &FallbackImage,
    ) -> Result<PreparedBindGroup<Self>, AsBindGroupError> {
        let images: Vec<_> = self
            .iter()
            .map(|handle| match images.get(handle) {
                Some(image) => image,
                None => fallback_image,
            })
            .collect();
        let texture_views: Vec<_> = images.iter().map(|image| &*image.texture_view).collect();
        let bindings = images
            .iter()
            .map(|image| OwnedBindingResource::TextureView(image.texture_view.clone()))
            .collect();

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureViewArray(&texture_views),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        Ok(PreparedBindGroup {
            bindings,
            bind_group,
            data: (),
        })
    }

    fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Blue Noise Texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(NOISE_TEXTURE_COUNT as u32),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
    }
}
