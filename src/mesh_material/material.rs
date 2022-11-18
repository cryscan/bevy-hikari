use super::{GpuStandardMaterial, GpuStandardMaterialBuffer, MeshMaterialSystems};
use bevy::{
    asset::{Asset, HandleId},
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp, RenderStage,
    },
    utils::{HashMap, HashSet},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

pub struct MaterialPlugin;
impl Plugin for MaterialPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedMaterials>()
                .init_resource::<MaterialRenderAssets>()
                .init_resource::<GpuStandardMaterials>()
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_material_assets.label(MeshMaterialSystems::PrepareAssets),
                );
        }
    }
}

#[derive(Default)]
pub struct GenericMaterialPlugin<M: Into<StandardMaterial>>(PhantomData<M>);

impl<M> Plugin for GenericMaterialPlugin<M>
where
    M: Into<StandardMaterial> + Clone + Asset,
{
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_system_to_stage(RenderStage::Extract, extract_material_assets::<M>);
        }
    }
}

#[derive(Default, Resource)]
pub struct MaterialRenderAssets {
    pub buffer: StorageBuffer<GpuStandardMaterialBuffer>,
    pub textures: BTreeSet<Handle<Image>>,
}

impl MaterialRenderAssets {
    pub fn add_standard_material_textures(&mut self, material: &StandardMaterial) {
        macro_rules! add_texture {
            ($texture_name:ident) => {
                if let Some(texture) = &material.$texture_name {
                    self.textures.insert(texture.clone_weak());
                }
            };
        }

        add_texture!(base_color_texture);
        add_texture!(emissive_texture);
        add_texture!(metallic_roughness_texture);
        add_texture!(normal_map_texture);
        add_texture!(occlusion_texture);
    }
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct GpuStandardMaterials(HashMap<HandleUntyped, (GpuStandardMaterial, u32)>);

#[derive(Default, Resource)]
pub struct ExtractedMaterials {
    extracted: Vec<(HandleUntyped, StandardMaterial)>,
    removed: Vec<HandleUntyped>,
}

fn extract_material_assets<M: Into<StandardMaterial> + Clone + Asset>(
    mut events: Extract<EventReader<AssetEvent<M>>>,
    assets: Extract<Res<Assets<M>>>,
    mut extracted_assets: ResMut<ExtractedMaterials>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                changed_assets.insert(handle);
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(&handle);
                removed.push(handle.clone_weak_untyped());
            }
        }
    }

    let mut extracted = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(material) = assets.get(handle) {
            let handle = handle.clone_weak_untyped();
            let material = material.clone().into();
            extracted.push((handle, material));
        }
    }

    extracted_assets.extracted.append(&mut extracted);
    extracted_assets.removed.append(&mut removed);
}

fn prepare_material_assets(
    mut extracted_assets: ResMut<ExtractedMaterials>,
    mut assets: Local<BTreeMap<HandleId, StandardMaterial>>,
    mut materials: ResMut<GpuStandardMaterials>,
    mut render_assets: ResMut<MaterialRenderAssets>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if extracted_assets.removed.is_empty() && extracted_assets.extracted.is_empty() {
        return;
    }

    for handle in extracted_assets.removed.drain(..) {
        materials.remove(&handle);
        assets.remove(&handle.into());
    }
    for (handle, material) in extracted_assets.extracted.drain(..) {
        render_assets.add_standard_material_textures(&material);
        assets.insert(handle.into(), material);
    }

    materials.clear();

    // TODO: remove unused textures.
    let textures: Vec<_> = render_assets.textures.iter().cloned().collect();
    let texture_id = |handle: &Option<Handle<Image>>| {
        if let Some(handle) = handle {
            match textures.binary_search(handle) {
                Ok(id) | Err(id) => id as u32,
            }
        } else {
            u32::MAX
        }
    };

    let materials = assets
        .iter()
        .enumerate()
        .map(|(offset, (handle, material))| {
            let handle = HandleUntyped::weak(*handle);

            let base_color = material.base_color.into();
            let base_color_texture = texture_id(&material.base_color_texture);

            let emissive = material.emissive.into();
            let emissive_texture = texture_id(&material.emissive_texture);

            let metallic_roughness_texture = texture_id(&material.metallic_roughness_texture);
            let normal_map_texture = texture_id(&material.normal_map_texture);
            let occlusion_texture = texture_id(&material.occlusion_texture);

            let material = GpuStandardMaterial {
                base_color,
                base_color_texture,
                emissive,
                emissive_texture,
                perceptual_roughness: material.perceptual_roughness,
                metallic: material.metallic,
                metallic_roughness_texture,
                reflectance: material.reflectance,
                normal_map_texture,
                occlusion_texture,
            };
            materials.insert(handle, (material, offset as u32));
            material
        })
        .collect();

    render_assets.buffer.get_mut().data = materials;
    render_assets
        .buffer
        .write_buffer(&render_device, &render_queue);
}
