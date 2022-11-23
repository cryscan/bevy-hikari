use super::{
    material::GpuStandardMaterials, mesh::GpuMeshes, GpuAliasEntry, GpuAliasTableBuffer,
    GpuEmissive, GpuEmissiveBuffer, GpuMesh, GpuStandardMaterial, MeshMaterialSystems,
};
use crate::{
    mesh_material::{GpuInstance, GpuInstanceBuffer, GpuNode, GpuNodeBuffer},
    transform::GlobalTransformQueue,
};
use bevy::{
    asset::Asset,
    ecs::query::QueryItem,
    math::{Vec3A, Vec4Swizzles},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
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
        app.add_plugin(ExtractComponentPlugin::<PreviousMeshUniform>::default())
            .add_plugin(UniformComponentPlugin::<PreviousMeshUniform>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedInstances>()
                .init_resource::<InstanceRenderAssets>()
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_instances
                        .label(MeshMaterialSystems::PrepareInstances)
                        .after(MeshMaterialSystems::PrepareAssets),
                );
        }
    }
}

#[derive(Default)]
pub struct GenericInstancePlugin<M: Into<StandardMaterial>>(PhantomData<M>);

impl<M> Plugin for GenericInstancePlugin<M>
where
    M: Into<StandardMaterial> + Asset,
{
    fn build(&self, app: &mut App) {
        app.add_event::<InstanceEvent<M>>().add_system_to_stage(
            CoreStage::PostUpdate,
            instance_event_system::<M>
                .after(TransformSystem::TransformPropagate)
                .after(VisibilitySystems::VisibilityPropagate)
                .after(VisibilitySystems::CalculateBounds),
        );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_system_to_stage(RenderStage::Extract, extract_instances::<M>);
        }
    }
}

#[derive(Default, Resource)]
pub struct InstanceRenderAssets {
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub instance_node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub emissive_buffer: StorageBuffer<GpuEmissiveBuffer>,
    pub emissive_node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub alias_table_buffer: StorageBuffer<GpuAliasTableBuffer>,
    pub instance_indices: DynamicUniformBuffer<InstanceIndex>,
}

impl InstanceRenderAssets {
    pub fn set(
        &mut self,
        instances: Vec<GpuInstance>,
        instance_nodes: Vec<GpuNode>,
        emissives: Vec<GpuEmissive>,
        emissive_nodes: Vec<GpuNode>,
        alias_table: Vec<GpuAliasEntry>,
    ) {
        self.instance_buffer.get_mut().data = instances;
        self.emissive_buffer.get_mut().data = emissives;
        self.alias_table_buffer.get_mut().data = alias_table;

        self.instance_node_buffer.get_mut().count = instance_nodes.len() as u32;
        self.instance_node_buffer.get_mut().data = instance_nodes;

        self.emissive_node_buffer.get_mut().count = emissive_nodes.len() as u32;
        self.emissive_node_buffer.get_mut().data = emissive_nodes;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.instance_node_buffer.write_buffer(device, queue);
        self.emissive_buffer.write_buffer(device, queue);
        self.emissive_node_buffer.write_buffer(device, queue);
        self.instance_indices.write_buffer(device, queue);
        self.alias_table_buffer.write_buffer(device, queue);
    }
}

#[derive(Default, Component, Clone, ShaderType)]
pub struct PreviousMeshUniform {
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
}

impl ExtractComponent for PreviousMeshUniform {
    type Query = &'static GlobalTransformQueue;
    type Filter = With<Handle<Mesh>>;
    type Out = Self;

    fn extract_component(queue: QueryItem<Self::Query>) -> Option<Self::Out> {
        let transform = queue[1];
        Some(PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        })
    }
}

pub enum InstanceEvent<M: Into<StandardMaterial> + Asset> {
    Created(Entity, Handle<Mesh>, Handle<M>, ComputedVisibility),
    Modified(Entity, Handle<Mesh>, Handle<M>, ComputedVisibility),
    Removed(Entity),
}

#[allow(clippy::type_complexity)]
fn instance_event_system<M: Into<StandardMaterial> + Asset>(
    mut events: EventWriter<InstanceEvent<M>>,
    removed: RemovedComponents<Handle<Mesh>>,
    mut set: ParamSet<(
        Query<
            (Entity, &Handle<Mesh>, &Handle<M>, &ComputedVisibility),
            Or<(Added<Handle<Mesh>>, Added<Handle<M>>)>,
        >,
        Query<
            (Entity, &Handle<Mesh>, &Handle<M>, &ComputedVisibility),
            Or<(
                Changed<GlobalTransform>,
                Changed<Handle<Mesh>>,
                Changed<Handle<M>>,
                Changed<ComputedVisibility>,
            )>,
        >,
    )>,
) {
    for entity in removed.iter() {
        events.send(InstanceEvent::Removed(entity));
    }
    for (entity, mesh, material, visibility) in &set.p0() {
        events.send(InstanceEvent::Created(
            entity,
            mesh.clone_weak(),
            material.clone_weak(),
            visibility.clone(),
        ));
    }
    for (entity, mesh, material, visibility) in &set.p1() {
        events.send(InstanceEvent::Modified(
            entity,
            mesh.clone_weak(),
            material.clone_weak(),
            visibility.clone(),
        ));
    }
}

#[allow(clippy::type_complexity)]
#[derive(Default, Resource)]
pub struct ExtractedInstances {
    extracted: Vec<(
        Entity,
        Aabb,
        GlobalTransform,
        Handle<Mesh>,
        HandleUntyped,
        ComputedVisibility,
    )>,
    removed: Vec<Entity>,
}

fn extract_instances<M: Into<StandardMaterial> + Asset>(
    mut events: Extract<EventReader<InstanceEvent<M>>>,
    query: Extract<Query<(&Aabb, &GlobalTransform)>>,
    mut extracted_instances: ResMut<ExtractedInstances>,
) {
    let mut extracted = vec![];
    let mut removed = vec![];

    for event in events.iter() {
        match event {
            InstanceEvent::Created(entity, mesh, material, visibility)
            | InstanceEvent::Modified(entity, mesh, material, visibility) => {
                if let Ok((aabb, transform)) = query.get(*entity) {
                    extracted.push((
                        *entity,
                        aabb.clone(),
                        *transform,
                        mesh.clone_weak(),
                        material.clone_weak_untyped(),
                        visibility.clone(),
                    ));
                }
            }
            InstanceEvent::Removed(entity) => removed.push(*entity),
        }
    }

    extracted_instances.extracted.append(&mut extracted);
    extracted_instances.removed.append(&mut removed);
}

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

#[derive(Component, Default, Clone, Copy)]
pub struct DynamicInstanceIndex(pub u32);

type Instances = BTreeMap<
    Entity,
    (
        GpuInstance,
        GpuMesh,
        GpuStandardMaterial,
        ComputedVisibility,
    ),
>;

/// Note: this system must run AFTER [`prepare_mesh_assets`].
#[allow(clippy::too_many_arguments)]
fn prepare_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut render_assets: ResMut<InstanceRenderAssets>,
    mut extracted_instances: ResMut<ExtractedInstances>,
    mut collection: Local<Instances>,
    meshes: Res<GpuMeshes>,
    materials: Res<GpuStandardMaterials>,
) {
    let instance_changed =
        !extracted_instances.extracted.is_empty() || !extracted_instances.removed.is_empty();

    for removed in extracted_instances.removed.drain(..) {
        collection.remove(&removed);
    }

    let mut prepare_next_frame = vec![];

    for (entity, aabb, transform, mesh, material, visibility) in extracted_instances
        .extracted
        .drain(..)
        .filter_map(|(entity, aabb, transform, mesh, material, visibility)| {
            match (meshes.get(&mesh), materials.get(&material)) {
                (Some(mesh), Some(material)) => {
                    Some((entity, aabb, transform, mesh, material, visibility))
                }
                _ => {
                    prepare_next_frame.push((entity, aabb, transform, mesh, material, visibility));
                    None
                }
            }
        })
    {
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

        // Note that the `GpuInstance` is partially constructed:
        // since node index is unknown at this point.
        let min = Vec3::from(min);
        let max = Vec3::from(max);
        collection.insert(
            entity,
            (
                GpuInstance {
                    min,
                    max,
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                    mesh: mesh.1,
                    material: material.1,
                    ..Default::default()
                },
                mesh.0.clone(),
                material.0.clone(),
                visibility,
            ),
        );
    }

    extracted_instances
        .extracted
        .append(&mut prepare_next_frame);

    // Since entities are cleared every frame, this should always be called.
    let mut add_instance_indices = |instances: &Instances| {
        render_assets.instance_indices.clear();
        let command_batch: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(id, (entity, (instance, _, _, _)))| {
                let component = InstanceIndex {
                    instance: id as u32,
                    material: instance.material,
                };
                let index = render_assets.instance_indices.push(component);
                (*entity, (DynamicInstanceIndex(index),))
            })
            .collect();
        commands.insert_or_spawn_batch(command_batch);
    };

    if instance_changed || meshes.is_changed() || materials.is_changed() {
        // Important: update mesh and material info for every instance
        let mut emissives = vec![];
        let mut alias_table = vec![];

        collection.retain(|_, (_, _, _, visibility)| visibility.is_visible_in_hierarchy());

        let mut instances: Vec<_> = collection
            .values()
            .map(|(instance, _, _, _)| instance)
            .cloned()
            .collect();

        let instance_nodes = match collection.is_empty() {
            true => vec![],
            false => {
                let bvh = BVH::build(&mut instances);
                bvh.flatten_custom(&GpuNode::pack)
            }
        };

        for ((instance, _, _, _), value) in collection.values_mut().zip_eq(instances.iter()) {
            // Assign the computed BVH node index, and mesh/material indices.
            *instance = value.clone();
        }

        add_instance_indices(&collection);

        for (id, (_, (instance, mesh, material, _))) in collection.iter().enumerate() {
            let emissive = material.emissive;
            let intensity = emissive.w * emissive.xyz().length();
            if intensity > 0.0 {
                // Compute alias table for light sampling
                let alias_table = {
                    let mut instance_table = mesh.build_alias_table(instance.transform);
                    let index = UVec2::new(alias_table.len() as u32, instance_table.len() as u32);
                    alias_table.append(&mut instance_table);
                    index
                };

                let surface_area = mesh
                    .transformed_primitive_areas(instance.transform)
                    .iter()
                    .sum();

                // Add to emissive list.
                let position = 0.5 * (instance.max + instance.min);
                let radius = 0.5 * (instance.max - instance.min).length();
                emissives.push(GpuEmissive {
                    emissive,
                    position,
                    radius,
                    instance: id as u32,
                    alias_table,
                    surface_area,
                    node_index: 0,
                });
            }
        }

        let emissive_nodes = match emissives.is_empty() {
            true => vec![],
            false => {
                let bvh = BVH::build(&mut emissives);
                bvh.flatten_custom(&GpuNode::pack)
            }
        };

        render_assets.set(
            instances,
            instance_nodes,
            emissives,
            emissive_nodes,
            alias_table,
        );
        render_assets.write_buffer(&render_device, &render_queue);
    } else {
        add_instance_indices(&collection);
        render_assets
            .instance_indices
            .write_buffer(&render_device, &render_queue);
    }
}
