use bevy::{
    pbr::{MeshPipeline, ShadowPipelineKey, SHADOW_FORMAT},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayout,
        render_resource::{
            BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
            BufferBindingType, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
            DepthStencilState, Extent3d, FragmentState, FrontFace, MultisampleState, PolygonMode,
            PrimitiveState, RenderPipelineDescriptor, ShaderStages, ShaderType,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
            StencilFaceState, StencilState, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, VertexState,
        },
        renderer::RenderDevice,
        view::ViewUniform,
        RenderApp,
    },
    window::WindowResized,
};

use crate::{image::DepthImage, PREPASS_SHADER_HANDLE};

pub struct PrepassPlugin;
impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PrepassTextures>()
            .add_system(resize_prepass_textures);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<PrepassPipeline>()
                .init_resource::<SpecializedMeshPipelines<PrepassPipeline>>();
        }
    }
}

pub struct PrepassTextures {
    pub depth: Handle<DepthImage>,
    pub normal: Handle<Image>,
}

impl FromWorld for PrepassTextures {
    fn from_world(world: &mut World) -> Self {
        let window = world.resource::<Windows>().primary();
        let size = Extent3d {
            width: window.width() as u32,
            height: window.height() as u32,
            depth_or_array_layers: 1,
        };

        let mut images = world.resource_mut::<Assets<DepthImage>>();
        let mut image = Image {
            texture_descriptor: TextureDescriptor {
                label: None,
                size,
                dimension: TextureDimension::D2,
                format: SHADOW_FORMAT,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            },
            ..Default::default()
        };
        image.resize(size);
        let depth = images.add(image.into());

        let mut images = world.resource_mut::<Assets<Image>>();
        let mut image = Image {
            texture_descriptor: TextureDescriptor {
                label: None,
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT,
            },
            ..Default::default()
        };
        image.resize(size);
        let normal = images.add(image);

        Self { depth, normal }
    }
}

pub struct PrepassPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
}

impl FromWorld for PrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
        });

        let mesh_pipeline = world.resource::<MeshPipeline>();
        let mesh_layout = mesh_pipeline.mesh_layout.clone();

        Self {
            view_layout,
            mesh_layout,
        }
    }
}

impl SpecializedMeshPipeline for PrepassPipeline {
    type Key = ShadowPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        ];
        let bind_group_layout = vec![self.view_layout.clone(), self.mesh_layout.clone()];

        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;

        Ok(RenderPipelineDescriptor {
            label: None,
            layout: Some(bind_group_layout),
            vertex: VertexState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
        })
    }
}

fn resize_prepass_textures(
    mut window_resized_events: EventReader<WindowResized>,
    windows: Res<Windows>,
    mut depth_images: ResMut<Assets<DepthImage>>,
    mut images: ResMut<Assets<Image>>,
    prepass_textures: Res<PrepassTextures>,
) {
    for event in window_resized_events.iter() {
        if event.id == windows.primary().id() {
            let size = Extent3d {
                width: event.width as u32,
                height: event.height as u32,
                depth_or_array_layers: 1,
            };
            if let Some(image) = depth_images.get_mut(&prepass_textures.depth) {
                image.resize(size);
            }
            if let Some(image) = images.get_mut(&prepass_textures.normal) {
                image.resize(size);
            }
        }
    }
}
