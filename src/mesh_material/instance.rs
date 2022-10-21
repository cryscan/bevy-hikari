use super::{
    material::GpuStandardMaterials, mesh::GpuMeshes, GpuLightSource, GpuLightSourceBuffer,
    MeshMaterialSystems,
};
use crate::{
    mesh_material::{GpuInstance, GpuInstanceBuffer, GpuNode, GpuNodeBuffer, IntoStandardMaterial},
    transform::GlobalTransformQueue,
    HikariConfig,
};
use bevy::{
    math::{Vec3A, Vec4Swizzles},
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
                .init_resource::<GpuLightSources>()
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
                .after(VisibilitySystems::VisibilityPropagate)
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
    pub source_buffer: StorageBuffer<GpuLightSourceBuffer>,
    pub instance_indices: DynamicUniformBuffer<InstanceIndex>,
}

impl InstanceRenderAssets {
    pub fn set(
        &mut self,
        instances: Vec<GpuInstance>,
        nodes: Vec<GpuNode>,
        sources: Vec<GpuLightSource>,
    ) {
        self.instance_buffer.get_mut().data = instances;

        self.node_buffer.get_mut().count = nodes.len() as u32;
        self.node_buffer.get_mut().data = nodes;

        self.source_buffer.get_mut().count = sources.len() as u32;
        self.source_buffer.get_mut().data = sources;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
        self.source_buffer.write_buffer(device, queue);
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
    query: Extract<Query<(Entity, &GlobalTransformQueue), With<Handle<Mesh>>>>,
) {
    for (entity, queue) in query.iter() {
        let transform = queue[1];
        let uniform = PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        };
        commands.get_or_spawn(entity).insert(uniform);
    }
}

#[derive(Default, Deref, DerefMut)]
pub struct GpuInstances(
    BTreeMap<Entity, (GpuInstance, Handle<Mesh>, HandleUntyped, ComputedVisibility)>,
);

#[derive(Default, Deref, DerefMut)]
pub struct GpuLightSources(BTreeMap<Entity, GpuLightSource>);

pub enum InstanceEvent<M: IntoStandardMaterial> {
    Created(Entity, Handle<Mesh>, Handle<M>, ComputedVisibility),
    Modified(Entity, Handle<Mesh>, Handle<M>, ComputedVisibility),
    Removed(Entity),
}

#[allow(clippy::type_complexity)]
fn instance_event_system<M: IntoStandardMaterial>(
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
pub struct ExtractedInstances<M: IntoStandardMaterial> {
    extracted: Vec<(
        Entity,
        Aabb,
        GlobalTransform,
        Handle<Mesh>,
        Handle<M>,
        ComputedVisibility,
    )>,
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
            InstanceEvent::Created(entity, mesh, material, visibility)
            | InstanceEvent::Modified(entity, mesh, material, visibility) => {
                if let Ok((aabb, transform)) = query.get(*entity) {
                    extracted.push((
                        *entity,
                        aabb.clone(),
                        *transform,
                        mesh.clone_weak(),
                        material.clone_weak(),
                        visibility.clone(),
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
) {
    for removed in extracted_instances.removed.drain(..) {
        instances.remove(&removed);
    }

    for (entity, aabb, transform, mesh, material, visibility) in
        extracted_instances.extracted.drain(..)
    {
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

        // Note that here the `GpuInstance` is partially constructed
        let min = Vec3::from(min);
        let max = Vec3::from(max);
        instances.insert(
            entity,
            (
                GpuInstance {
                    min,
                    max,
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                    ..Default::default()
                },
                mesh,
                material,
                visibility,
            ),
        );
    }
}

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

#[derive(Component, Default, Clone, Copy)]
pub struct DynamicInstanceIndex(pub u32);

/// Note: this system must run AFTER [`prepare_mesh_assets`].
#[allow(clippy::too_many_arguments)]
fn prepare_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    config: Res<HikariConfig>,
    mut render_assets: ResMut<InstanceRenderAssets>,
    mut instances: ResMut<GpuInstances>,
    mut sources: ResMut<GpuLightSources>,
    meshes: Res<GpuMeshes>,
    materials: Res<GpuStandardMaterials>,
) {
    // Since entities are cleared every frame, this should always be called.
    let mut add_instance_indices = |instances: &GpuInstances| {
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

    if meshes.is_changed() || materials.is_changed() || instances.is_changed() {
        // Important: update mesh and material info for every instance
        sources.clear();
        instances.retain(|_, (instance, mesh, material, visibility)| {
            if !visibility.is_visible_in_hierarchy() {
                return false;
            }
            if let (Some(mesh), Some(material)) = (meshes.get(mesh), materials.get(material)) {
                instance.mesh = mesh.1;
                instance.material = material.1;
                true
            } else {
                false
            }
        });

        let mut values: Vec<_> = instances
            .values()
            .map(|(instance, _, _, _)| instance)
            .cloned()
            .collect();

        let nodes = if !instances.is_empty() {
            let bvh = BVH::build(&mut values);
            bvh.flatten_custom(&GpuNode::pack)
        } else {
            Vec::new()
        };

        for (instance, value) in instances.values_mut().zip_eq(values.iter()) {
            instance.0 = *value;
        }

        add_instance_indices(&instances);

        for (id, (entity, (instance, _, material, _))) in instances.iter().enumerate() {
            if let Some(material) = materials.get(material) {
                // Add it to the light source list if it's emissive.
                let emissive = material.0.emissive;
                if emissive.w * emissive.xyz().length() > config.emissive_threshold {
                    let position = 0.5 * (instance.max + instance.min);
                    let radius = 0.5 * (instance.max - instance.min).length();
                    sources.insert(
                        *entity,
                        GpuLightSource {
                            emissive,
                            position,
                            radius,
                            instance: id as u32,
                        },
                    );
                }
            }
        }
        let sources = sources.values().cloned().collect();

        render_assets.set(values, nodes, sources);
        render_assets.write_buffer(&render_device, &render_queue);
    } else {
        add_instance_indices(&instances);
        render_assets
            .instance_indices
            .write_buffer(&render_device, &render_queue);
    }
}
