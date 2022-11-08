use crate::{
    light::{LightPassNode, LightPlugin},
    mesh_material::MeshMaterialPlugin,
    overlay::{OverlayPassNode, OverlayPlugin},
    post_process::{PostProcessPassNode, PostProcessPlugin},
    prepass::{PrepassNode, PrepassPlugin},
    transform::TransformPlugin,
    view::ViewPlugin,
};
use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::MainPass3dNode,
    ecs::query::QueryItem,
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{RenderGraph, SlotInfo, SlotType},
        render_resource::*,
        renderer::RenderDevice,
        texture::{CompressedImageFormats, FallbackImage, ImageType},
        RenderApp,
    },
};
use std::num::NonZeroU32;

#[macro_use]
extern crate num_derive;

pub mod light;
pub mod mesh_material;
pub mod overlay;
pub mod post_process;
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
        pub const POST_PROCESS_PASS: &str = "post_process_pass";
        pub const OVERLAY_PASS: &str = "overlay_pass";
    }
}

pub const WORKGROUP_SIZE: u32 = 8;
pub const NOISE_TEXTURE_COUNT: usize = 16;

pub const UTILS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4462033275253590181);
pub const MESH_VIEW_TYPES_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 10086770709483722043);
pub const MESH_VIEW_BINDINGS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 8835349515886344623);
pub const MESH_MATERIAL_TYPES_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 15819591594687298858);
pub const MESH_MATERIAL_BINDINGS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5025976374517268);
pub const DEFERRED_BINDINGS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14467895678105108252);
pub const RESERVOIR_TYPES_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 7770589395703787378);
pub const RESERVOIR_BINDINGS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 11658053183743104810);
pub const RESERVOIR_FUNCTIONS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 7650021494161056224);
pub const PREPASS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4693612430004931427);
pub const LIGHT_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 9657319286592943583);
pub const DENOISE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 5179661212363325472);
pub const TONE_MAPPING_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 3567017338952956671);
pub const TAA_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 1780446804546284);
pub const SMAA_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 3793959332758430953);
pub const FSR1_EASU_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 11823787237582686663);
pub const FSR1_RCAS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 17003547378277520107);
pub const OVERLAY_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 10969344919103020615);
pub const QUAD_MESH_HANDLE: HandleUntyped =
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
            MESH_VIEW_TYPES_SHADER_HANDLE,
            "shaders/mesh_view_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_VIEW_BINDINGS_SHADER_HANDLE,
            "shaders/mesh_view_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_MATERIAL_TYPES_SHADER_HANDLE,
            "shaders/mesh_material_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_MATERIAL_BINDINGS_SHADER_HANDLE,
            "shaders/mesh_material_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DEFERRED_BINDINGS_SHADER_HANDLE,
            "shaders/deferred_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RESERVOIR_TYPES_SHADER_HANDLE,
            "shaders/reservoir_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RESERVOIR_BINDINGS_SHADER_HANDLE,
            "shaders/reservoir_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RESERVOIR_FUNCTIONS_SHADER_HANDLE,
            "shaders/reservoir_functions.wgsl",
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
            DENOISE_SHADER_HANDLE,
            "shaders/denoise.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            TONE_MAPPING_SHADER_HANDLE,
            "shaders/tone_mapping.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            TAA_SHADER_HANDLE,
            "shaders/taa.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SMAA_SHADER_HANDLE,
            "shaders/smaa.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            OVERLAY_SHADER_HANDLE,
            "shaders/overlay.wgsl",
            Shader::from_wgsl
        );

        let mut assets = app.world.resource_mut::<Assets<_>>();
        assets.set_untracked(
            FSR1_EASU_SHADER_HANDLE,
            Shader::from_spirv(include_bytes!("shaders/fsr/fsr_pass_easu.spv").as_ref()),
        );
        assets.set_untracked(
            FSR1_RCAS_SHADER_HANDLE,
            Shader::from_spirv(include_bytes!("shaders/fsr/fsr_pass_rcas.spv").as_ref()),
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
            .register_type::<Taa>()
            .register_type::<Upscale>()
            .add_plugin(ExtractResourcePlugin::<NoiseTextures>::default())
            .add_plugin(ExtractComponentPlugin::<HikariConfig>::default())
            .add_plugin(TransformPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshMaterialPlugin)
            .add_plugin(PrepassPlugin)
            .add_plugin(LightPlugin)
            .add_plugin(PostProcessPlugin)
            .add_plugin(OverlayPlugin)
            .add_startup_system(noise_load_system)
            .add_system_to_stage(CoreStage::PostUpdate, hikari_config_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let prepass_node = PrepassNode::new(&mut render_app.world);
            let light_pass_node = LightPassNode::new(&mut render_app.world);
            let post_process_pass_node = PostProcessPassNode::new(&mut render_app.world);
            let overlay_pass_node = OverlayPassNode::new(&mut render_app.world);

            let mut graph = render_app.world.resource_mut::<RenderGraph>();

            let mut hikari_graph = RenderGraph::default();
            hikari_graph.add_node(graph::node::PREPASS, prepass_node);
            hikari_graph.add_node(graph::node::LIGHT_PASS, light_pass_node);
            hikari_graph.add_node(graph::node::POST_PROCESS_PASS, post_process_pass_node);
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
                    graph::node::POST_PROCESS_PASS,
                    LightPassNode::IN_VIEW,
                )
                .unwrap();
            hikari_graph
                .add_node_edge(graph::node::LIGHT_PASS, graph::node::POST_PROCESS_PASS)
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
                .add_node_edge(graph::node::POST_PROCESS_PASS, graph::node::OVERLAY_PASS)
                .unwrap();
            graph.add_sub_graph(graph::NAME, hikari_graph);
        }
    }
}

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct HikariConfig {
    /// The interval of frames between sample validation passes.
    pub direct_validate_interval: usize,
    /// The interval of frames between sample validation passes.
    pub emissive_validate_interval: usize,
    /// Temporal reservoir sample count is capped by this value.
    pub max_temporal_reuse_count: usize,
    /// Spatial reservoir sample count is capped by this value.
    pub max_spatial_reuse_count: usize,
    /// Half angle of the solar cone apex in radians.
    pub solar_angle: f32,
    /// Count of indirect bounces.
    pub indirect_bounces: usize,
    /// Threshold for the indirect luminance to reduce fireflies.
    pub max_indirect_luminance: f32,
    /// Clear color override.
    pub clear_color: Color,
    /// Whether to do temporal sample reuse in ReSTIR.
    pub temporal_reuse: bool,
    /// Whether to do spatial sample reuse in ReSTIR.
    pub spatial_reuse: bool,
    /// Whether to do noise filtering.
    pub denoise: bool,
    /// Which temporal filtering implementation to use.
    pub taa: Taa,
    /// Which upscaling implementation to use.
    pub upscale: Upscale,
}

impl Default for HikariConfig {
    fn default() -> Self {
        Self {
            direct_validate_interval: 3,
            emissive_validate_interval: 5,
            max_temporal_reuse_count: 50,
            max_spatial_reuse_count: 800,
            solar_angle: 0.046,
            clear_color: Color::rgb(0.4, 0.4, 0.4),
            indirect_bounces: 1,
            max_indirect_luminance: 10.0,
            temporal_reuse: true,
            spatial_reuse: true,
            denoise: true,
            taa: Taa::default(),
            upscale: Upscale::default(),
        }
    }
}

impl ExtractComponent for HikariConfig {
    type Query = &'static Self;
    type Filter = ();

    fn extract_component(item: QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

fn hikari_config_system(
    mut commands: Commands,
    clear_color: Res<ClearColor>,
    query: Query<Entity, (With<Camera>, Without<HikariConfig>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(HikariConfig {
            clear_color: clear_color.0,
            ..Default::default()
        });
    }
}

/// Temporal Anti-Aliasing Method to use.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Reflect)]
pub enum Taa {
    #[default]
    Jasmine,
    None,
}

/// Upscale method to use.
#[derive(Debug, Clone, Copy, Reflect)]
pub enum Upscale {
    /// [AMD FidelityFX™ Super Resolution](https://gpuopen.com/fidelityfx-superresolution/).
    Fsr1 {
        /// Renders the main pass and post process on a low resolution texture.
        ratio: f32,
        /// From 0.0 - 2.0 where 0.0 means max sharpness.
        sharpness: f32,
    },
    /// [Filmic SMAA TU4x](https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf).
    SmaaTu4x {
        /// Renders the main pass and post process on a low resolution texture.
        ratio: f32,
    },
    None,
}

impl Default for Upscale {
    fn default() -> Self {
        Self::SmaaTu4x { ratio: 1.0 }
    }
}

impl Upscale {
    pub fn ratio(&self) -> f32 {
        match self {
            Upscale::Fsr1 { ratio, .. } | Upscale::SmaaTu4x { ratio } => ratio.clamp(1.0, 2.0),
            Upscale::None => 1.0,
        }
    }

    pub fn sharpness(&self) -> f32 {
        match self {
            Upscale::Fsr1 { sharpness, .. } => *sharpness,
            _ => 0.0,
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
