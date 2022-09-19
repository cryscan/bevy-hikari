use super::{
    GpuStandardMaterial, GpuStandardMaterialBuffer, IntoStandardMaterial, MeshMaterialSystems,
};
use bevy::{
    asset::HandleId,
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
                .init_resource::<MaterialRenderAssets>()
                .init_resource::<StandardMaterials>()
                .init_resource::<GpuStandardMaterials>()
                .init_resource::<GpuStandardMaterialOffsets>()
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_material_assets
                        .label(MeshMaterialSystems::PrepareAssets)
                        .after(MeshMaterialSystems::PrePrepareAssets),
                );
        }
    }
}

#[derive(Default)]
pub struct GenericMaterialPlugin<M: IntoStandardMaterial>(PhantomData<M>);
impl<M: IntoStandardMaterial> Plugin for GenericMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_system_to_stage(RenderStage::Extract, extract_material_assets::<M>)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_generic_material_assets::<M>
                        .label(MeshMaterialSystems::PrePrepareAssets),
                );
        }
    }
}

#[derive(Default)]
pub struct MaterialRenderAssets {
    pub buffer: StorageBuffer<GpuStandardMaterialBuffer>,
    pub textures: BTreeSet<Handle<Image>>,
}

#[derive(Default, Deref, DerefMut)]
pub struct StandardMaterials(BTreeMap<HandleId, bevy::pbr::StandardMaterial>);

#[derive(Default, Deref, DerefMut)]
pub struct GpuStandardMaterials(HashMap<HandleUntyped, GpuStandardMaterial>);

#[derive(Default, Deref, DerefMut)]
pub struct GpuStandardMaterialOffsets(HashMap<HandleUntyped, u32>);

#[derive(Default)]
pub struct ExtractedMaterials<M: IntoStandardMaterial> {
    extracted: Vec<(Handle<M>, M)>,
    removed: Vec<Handle<M>>,
}

fn extract_material_assets<M: IntoStandardMaterial>(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<M>>>,
    assets: Extract<Res<Assets<M>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                changed_assets.insert(handle.clone_weak());
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut extracted = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(material) = assets.get(&handle) {
            extracted.push((handle, material.clone()));
        }
    }

    commands.insert_resource(ExtractedMaterials { extracted, removed });
}

fn prepare_generic_material_assets<M: IntoStandardMaterial>(
    mut extracted_assets: ResMut<ExtractedMaterials<M>>,
    mut materials: ResMut<StandardMaterials>,
    render_assets: ResMut<MaterialRenderAssets>,
) {
    for handle in extracted_assets.removed.drain(..) {
        materials.remove(&handle.id);
    }

    let render_assets = render_assets.into_inner();
    for (handle, material) in extracted_assets.extracted.drain(..) {
        let material = material.into_standard_material(render_assets);
        materials.insert(handle.id, material);
    }
}

fn prepare_material_assets(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    materials: Res<StandardMaterials>,
    mut offsets: ResMut<GpuStandardMaterialOffsets>,
    mut assets: ResMut<GpuStandardMaterials>,
    mut render_assets: ResMut<MaterialRenderAssets>,
) {
    if !materials.is_changed() {
        return;
    }

    assets.clear();
    offsets.clear();

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

    let materials = materials
        .iter()
        .enumerate()
        .map(|(offset, (handle, material))| {
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

            let handle = HandleUntyped::weak(*handle);
            assets.insert(handle.clone(), material.clone());
            offsets.insert(handle, offset as u32);
            material
        })
        .collect();

    render_assets.buffer.get_mut().data = materials;
    render_assets
        .buffer
        .write_buffer(&render_device, &render_queue);
}
