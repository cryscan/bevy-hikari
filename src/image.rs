use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::TextureViewDescriptor,
        renderer::RenderDevice,
        texture::{DefaultImageSampler, GpuImage, ImageSampler},
    },
};

pub struct ImagePlugin;
impl Plugin for ImagePlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<DepthImage>()
            .add_plugin(RenderAssetPlugin::<DepthImage>::default());
    }
}

#[derive(Debug, Default, Clone, TypeUuid, Deref, DerefMut)]
#[uuid = "e3e32e03-16b3-4a0a-a605-b8428cf9482c"]
pub struct DepthImage(Image);

impl From<Image> for DepthImage {
    fn from(image: Image) -> Self {
        Self(image)
    }
}

impl RenderAsset for DepthImage {
    type ExtractedAsset = Image;
    type PreparedAsset = GpuImage;
    type Param = (SRes<RenderDevice>, SRes<DefaultImageSampler>);

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.0.clone()
    }

    fn prepare_asset(
        image: Self::ExtractedAsset,
        (render_device, default_sampler): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let texture = render_device.create_texture(&image.texture_descriptor);

        let texture_view = texture.create_view(
            image
                .texture_view_descriptor
                .or_else(|| Some(TextureViewDescriptor::default()))
                .as_ref()
                .unwrap(),
        );
        let size = Vec2::new(
            image.texture_descriptor.size.width as f32,
            image.texture_descriptor.size.height as f32,
        );
        let sampler = match image.sampler_descriptor {
            ImageSampler::Default => (***default_sampler).clone(),
            ImageSampler::Descriptor(descriptor) => render_device.create_sampler(&descriptor),
        };

        Ok(GpuImage {
            texture,
            texture_view,
            texture_format: image.texture_descriptor.format,
            sampler,
            size,
        })
    }
}
