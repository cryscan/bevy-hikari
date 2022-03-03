use crate::OVERLAY_SHADER_HANDLE;
use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::MaterialPipeline,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssets},
        render_resource::*,
        renderer::RenderDevice,
    },
};

#[derive(Debug, Clone, TypeUuid)]
#[uuid = "3eb25222-95fd-11ec-b909-0242ac120002"]
pub struct OverlayMaterial(pub Handle<Image>);

#[derive(Clone)]
pub struct GpuOverlayMaterial {
    bind_group: BindGroup,
}

impl RenderAsset for OverlayMaterial {
    type ExtractedAsset = OverlayMaterial;
    type PreparedAsset = GpuOverlayMaterial;
    type Param = (
        SRes<RenderDevice>,
        SRes<MaterialPipeline<Self>>,
        SRes<RenderAssets<Image>>,
    );
    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (render_device, material_pipeline, images): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let image = if let Some(result) = images.get(&material.0) {
            result
        } else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&image.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&image.sampler),
                },
            ],
            label: Some("overlay_bind_group"),
            layout: &material_pipeline.material_layout,
        });

        Ok(GpuOverlayMaterial { bind_group })
    }
}

impl Material for OverlayMaterial {
    fn vertex_shader(_asset_server: &AssetServer) -> Option<Handle<Shader>> {
        Some(OVERLAY_SHADER_HANDLE.typed())
    }

    fn fragment_shader(_asset_server: &AssetServer) -> Option<Handle<Shader>> {
        Some(OVERLAY_SHADER_HANDLE.typed())
    }

    fn bind_group(render_asset: &<Self as RenderAsset>::PreparedAsset) -> &BindGroup {
        &render_asset.bind_group
    }

    fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("overlay_layout"),
        })
    }

    fn alpha_mode(_material: &<Self as RenderAsset>::PreparedAsset) -> AlphaMode {
        AlphaMode::Blend
    }
}
