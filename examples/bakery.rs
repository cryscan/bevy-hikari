use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        mesh::{PrimitiveTopology, VertexAttributeValues},
        primitives::Aabb,
    },
    transform::TransformSystem,
};
use itertools::Itertools;
use std::{collections::BTreeMap, f32::consts::PI};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<TriangleTable>()
        .add_startup_system(setup)
        .add_system_to_stage(
            CoreStage::PostUpdate,
            update_triangle_table.after(TransformSystem::TransformPropagate),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Plane
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 5.0 })),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..Default::default()
    });
    // Cube
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube::default())),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..Default::default()
    });

    // Only directional light is supported
    const HALF_SIZE: f32 = 5.0;
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
            shadow_projection: OrthographicProjection {
                left: -HALF_SIZE,
                right: HALF_SIZE,
                bottom: -HALF_SIZE,
                top: HALF_SIZE,
                near: -10.0 * HALF_SIZE,
                far: 10.0 * HALF_SIZE,
                ..Default::default()
            },
            shadows_enabled: true,
            ..Default::default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, -PI / 8.0, -PI / 4.0, 0.0),
            ..Default::default()
        },
        ..Default::default()
    });

    // Camera
    commands.spawn_bundle(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

#[derive(Default, Deref, DerefMut)]
struct TriangleTable(BTreeMap<Entity, Vec<Triangle>>);

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct Triangle([Vec3A; 3]);

impl Triangle {
    pub fn aabb(&self) -> Aabb {
        let min = self.iter().copied().reduce(Vec3A::min).unwrap();
        let max = self.iter().copied().reduce(Vec3A::max).unwrap();
        Aabb::from_min_max(min.into(), max.into())
    }

    pub fn centroid(&self) -> Vec3A {
        (self[0] + self[1] + self[2]) / 3.0
    }
}

#[allow(clippy::type_complexity)]
fn update_triangle_table(
    meshes: Res<Assets<Mesh>>,
    mut triangle_table: ResMut<TriangleTable>,
    query: Query<
        (Entity, &Handle<Mesh>, &GlobalTransform),
        Or<(Added<Handle<Mesh>>, Changed<GlobalTransform>)>,
    >,
) {
    for (entity, mesh, transform) in query.iter() {
        if let Some(mesh) = meshes.get(mesh) {
            if let Some((vertices, indices)) = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(VertexAttributeValues::as_float3)
                .zip(mesh.indices())
            {
                let transform = transform.compute_matrix();
                let mut triangles = vec![];

                match mesh.primitive_topology() {
                    PrimitiveTopology::TriangleList => {
                        for mut chunk in &indices.iter().chunks(3) {
                            let (v0, v1, v2) = chunk.next_tuple().unwrap();
                            let vertices = [v0, v1, v2]
                                .map(|id| vertices[id])
                                .map(|[x, y, z]| transform.transform_point3a(Vec3A::new(x, y, z)));
                            triangles.push(Triangle(vertices));
                        }
                    }
                    PrimitiveTopology::TriangleStrip => {
                        for (i, (v0, v1, v2)) in indices.iter().tuple_windows().enumerate() {
                            let vertices = if i % 2 == 0 {
                                [v0, v1, v2]
                            } else {
                                [v1, v0, v2]
                            }
                            .map(|id| vertices[id])
                            .map(|[x, y, z]| transform.transform_point3a(Vec3A::new(x, y, z)));
                            triangles.push(Triangle(vertices));
                        }
                    }
                    _ => (),
                }

                triangle_table.insert(entity, triangles);
            }
        }
    }
}
