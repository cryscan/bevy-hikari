use super::{
    material::GpuStandardMaterials,
    mesh::{GpuMeshes, MeshAssetState},
    MeshMaterialSystems,
};
use crate::{
    mesh::{GpuInstance, GpuInstanceBuffer, GpuNode, GpuNodeBuffer, IntoStandardMaterial},
    transform::PreviousGlobalTransform,
};
use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        extract_component::UniformComponentPlugin,
        primitives::Aabb,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        view::VisibilitySystems,
        Extract, RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use bvh::bvh::BVH;
use itertools::Itertools;
use std::{collections::BTreeMap, marker::PhantomData};

pub struct InstancePlugin;
impl Plugin for InstancePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(UniformComponentPlugin::<PreviousMeshUniform>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<GpuInstances>()
                .init_resource::<InstanceRenderAssets>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_transforms)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_instances
                        .label(MeshMaterialSystems::PostPrepareInstances)
                        .after(MeshMaterialSystems::PrepareInstances),
                );
        }
    }
}

#[derive(Default)]
pub struct GenericInstancePlugin<M: IntoStandardMaterial>(PhantomData<M>);
impl<M: IntoStandardMaterial> Plugin for GenericInstancePlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_event::<InstanceEvent<M>>().add_system_to_stage(
            CoreStage::PostUpdate,
            instance_event_system::<M>
                .after(TransformSystem::TransformPropagate)
                .after(VisibilitySystems::CalculateBounds),
        );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_system_to_stage(RenderStage::Extract, extract_instances::<M>)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_generic_instances::<M>
                        .label(MeshMaterialSystems::PrepareInstances)
                        .after(MeshMaterialSystems::PrepareAssets),
                );
        }
    }
}

#[derive(Default)]
pub struct InstanceRenderAssets {
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub instance_indices: DynamicUniformBuffer<InstanceIndex>,
}

impl InstanceRenderAssets {
    pub fn set(&mut self, instances: Vec<GpuInstance>, nodes: Vec<GpuNode>) {
        self.instance_buffer.get_mut().data = instances;
        self.node_buffer.get_mut().count = nodes.len() as u32;
        self.node_buffer.get_mut().data = nodes;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
        self.instance_indices.write_buffer(device, queue);
    }
}

#[derive(Default, Component, Clone, ShaderType)]
pub struct PreviousMeshUniform {
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
}

#[allow(clippy::type_complexity)]
fn extract_mesh_transforms(
    mut commands: Commands,
    query: Extract<Query<(Entity, &PreviousGlobalTransform), With<Handle<Mesh>>>>,
) {
    for (entity, transform) in query.iter() {
        let transform = transform.compute_matrix();
        let uniform = PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        };
        commands.get_or_spawn(entity).insert(uniform);
    }
}

#[derive(Default, Deref, DerefMut)]
pub struct GpuInstances(BTreeMap<Entity, GpuInstance>);

pub enum InstanceEvent<M: IntoStandardMaterial> {
    Created(Entity, Handle<Mesh>, Handle<M>),
    Modified(Entity, Handle<Mesh>, Handle<M>),
    Removed(Entity),
}

#[allow(clippy::type_complexity)]
fn instance_event_system<M: IntoStandardMaterial>(
    mut events: EventWriter<InstanceEvent<M>>,
    removed: RemovedComponents<Handle<Mesh>>,
    mut set: ParamSet<(
        Query<(Entity, &Handle<Mesh>, &Handle<M>), Or<(Added<Handle<Mesh>>, Added<Handle<M>>)>>,
        Query<
            (Entity, &Handle<Mesh>, &Handle<M>),
            Or<(
                Changed<Transform>,
                Changed<Handle<Mesh>>,
                Changed<Handle<M>>,
            )>,
        >,
    )>,
) {
    for entity in removed.iter() {
        events.send(InstanceEvent::Removed(entity));
    }
    for (entity, mesh, material) in &set.p0() {
        events.send(InstanceEvent::Created(
            entity,
            mesh.clone_weak(),
            material.clone_weak(),
        ));
    }
    for (entity, mesh, material) in &set.p1() {
        events.send(InstanceEvent::Modified(
            entity,
            mesh.clone_weak(),
            material.clone_weak(),
        ));
    }
}

pub struct ExtractedInstances<M: IntoStandardMaterial> {
    extracted: Vec<(Entity, Aabb, GlobalTransform, Handle<Mesh>, Handle<M>)>,
    removed: Vec<Entity>,
}

fn extract_instances<M: IntoStandardMaterial>(
    mut commands: Commands,
    mut events: Extract<EventReader<InstanceEvent<M>>>,
    query: Extract<Query<(&Aabb, &GlobalTransform)>>,
) {
    let mut extracted = vec![];
    let mut removed = vec![];

    for event in events.iter() {
        match event {
            InstanceEvent::Created(entity, mesh, material)
            | InstanceEvent::Modified(entity, mesh, material) => {
                if let Ok((aabb, transform)) = query.get(*entity) {
                    extracted.push((
                        *entity,
                        aabb.clone(),
                        transform.clone(),
                        mesh.clone_weak(),
                        material.clone_weak(),
                    ));
                }
            }
            InstanceEvent::Removed(entity) => removed.push(*entity),
        }
    }

    commands.insert_resource(ExtractedInstances { extracted, removed });
}

fn prepare_generic_instances<M: IntoStandardMaterial>(
    mut extracted_instances: ResMut<ExtractedInstances<M>>,
    mut instances: ResMut<GpuInstances>,
    meshes: Res<GpuMeshes>,
    materials: Res<GpuStandardMaterials>,
    asset_state: Res<MeshAssetState>,
) {
    if *asset_state == MeshAssetState::Dirty {
        panic!("Mesh assets must be prepared before instances!");
    }

    for removed in extracted_instances.removed.drain(..) {
        instances.remove(&removed);
    }
    for (entity, aabb, transform, mesh, material) in extracted_instances.extracted.drain(..) {
        let material = HandleUntyped::weak(material.id);
        let transform = transform.compute_matrix();
        let center = transform.transform_point3a(aabb.center);
        let vertices: Vec<_> = (0..8i32)
            .map(|index| {
                let x = 2 * (index & 1) - 1;
                let y = 2 * ((index >> 1) & 1) - 1;
                let z = 2 * ((index >> 2) & 1) - 1;
                let vertex = aabb.half_extents * Vec3A::new(x as f32, y as f32, z as f32);
                transform.transform_vector3a(vertex)
            })
            .collect();

        let mut min = Vec3A::ZERO;
        let mut max = Vec3A::ZERO;
        for vertex in vertices {
            min = min.min(vertex);
            max = max.max(vertex);
        }
        min += center;
        max += center;

        if let (Some(mesh), Some(material)) = (meshes.get(&mesh), materials.get(&material)) {
            let min = Vec3::from(min);
            let max = Vec3::from(max);
            let slice = mesh.1;
            let material = material.1;
            instances.insert(
                entity,
                GpuInstance {
                    min,
                    max,
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                    slice,
                    material,
                    node_index: 0,
                },
            );
        }
    }
}

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

#[derive(Component, Default, Clone, Copy)]
pub struct DynamicInstanceIndex(pub u32);

fn prepare_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut render_assets: ResMut<InstanceRenderAssets>,
    mut instances: ResMut<GpuInstances>,
    asset_state: Res<MeshAssetState>,
) {
    if *asset_state == MeshAssetState::Dirty {
        panic!("Mesh assets must be prepared before instances!");
    }

    if instances.is_empty() {
        return;
    }

    let mut add_instance_indices = |instances: &GpuInstances| {
        render_assets.instance_indices.clear();
        let command_batch: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(id, (entity, instance))| {
                let component = InstanceIndex {
                    instance: id as u32,
                    material: instance.material.value,
                };
                let index = render_assets.instance_indices.push(component);
                (*entity, (DynamicInstanceIndex(index),))
            })
            .collect();
        commands.insert_or_spawn_batch(command_batch);
    };

    if *asset_state != MeshAssetState::Clean || instances.is_changed() {
        let mut values: Vec<_> = instances.values().cloned().collect();
        let bvh = BVH::build(&mut values);

        for (instance, value) in instances.values_mut().zip_eq(values.iter()) {
            *instance = *value;
        }

        add_instance_indices(&instances);

        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            primitive_index,
        });
        render_assets.set(values, nodes);
        render_assets.write_buffer(&render_device, &render_queue);
    } else {
        add_instance_indices(&instances);
        render_assets
            .instance_indices
            .write_buffer(&render_device, &render_queue);
    }
}
