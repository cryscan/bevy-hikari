use bevy::{
    asset::load_internal_asset,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::RenderAssets, render_resource::*, renderer::RenderDevice,
        texture::FallbackImage, RenderApp, RenderStage,
    },
};

mod denoise;
mod nearest_velocity;
mod smaa;

pub const DEMODULATION_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 10334148353041544650);
pub const DENOISE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 7622960073196288673);
pub const MODULATION_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 9584218513855118856);
pub const NEAREST_VELOCITY_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 8932183074989906911);
pub const SMAA_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 1268657799090581450);
pub const TAA_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 21044543884618662);
pub const INVERSE_TOME_MAPPING_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 15280876093739544327);

pub struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DEMODULATION_SHADER_HANDLE,
            "../shaders/post_process/demodulation.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DENOISE_SHADER_HANDLE,
            "../shaders/post_process/denoise.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MODULATION_SHADER_HANDLE,
            "../shaders/post_process/modulation.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            NEAREST_VELOCITY_SHADER_HANDLE,
            "../shaders/post_process/nearest_velocity.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SMAA_SHADER_HANDLE,
            "../shaders/post_process/smaa.wgsl",
            Shader::from_wgsl
        );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<PostProcessSamplers>()
                .init_resource::<denoise::DenoisePipelines>()
                .init_resource::<nearest_velocity::NearestVelocityPipeline>()
                .init_resource::<smaa::SmaaPipelines>()
                .add_system_to_stage(RenderStage::Prepare, denoise::prepare_denoise_textures)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    nearest_velocity::prepare_nearest_velocity_texture,
                )
                .add_system_to_stage(RenderStage::Queue, queue_sampler_bind_group)
                .add_system_to_stage(RenderStage::Queue, denoise::queue_denoise_bind_groups);
        }
    }
}

#[derive(Resource)]
pub struct PostProcessSamplers {
    pub nearest_sampler: Sampler,
    pub linear_sampler: Sampler,
}

impl FromWorld for PostProcessSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        Self {
            nearest_sampler,
            linear_sampler,
        }
    }
}

impl AsBindGroup for PostProcessSamplers {
    type Data = ();

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        _: &RenderAssets<Image>,
        _: &FallbackImage,
    ) -> Result<PreparedBindGroup<Self>, AsBindGroupError> {
        let bindings = vec![
            OwnedBindingResource::Sampler(self.nearest_sampler.clone()),
            OwnedBindingResource::Sampler(self.linear_sampler.clone()),
        ];
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.nearest_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.linear_sampler),
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
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
}

#[derive(Resource)]
pub struct SamplerBindGroup(pub BindGroup);

fn queue_sampler_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    samplers: Res<PostProcessSamplers>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
) {
    let layout = PostProcessSamplers::bind_group_layout(&render_device);
    if let Ok(prepared_bind_group) =
        samplers.as_bind_group(&layout, &render_device, &images, &fallback_image)
    {
        commands.insert_resource(SamplerBindGroup(prepared_bind_group.bind_group));
    }
}
