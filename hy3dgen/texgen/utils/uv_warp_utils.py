# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.
import os

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh
import numpy as np


def mesh_uv_wrap(mesh, padding=2, resolution=1024, max_iterations=4):
    import xatlas

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    large_mesh_mode = False
    if len(mesh.faces) > 100000 and len(mesh.faces) < 200000:
        large_mesh_mode = True
        print("Warning: The mesh has more than 100,000 faces, which may cause slowdowns.")
    if len(mesh.faces) > 200000:
        raise ValueError("The mesh has more than 200,000 faces, which is not supported.")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()

    # Add the mesh to the atlas
    atlas.add_mesh(vertices, faces)

    chart_options = xatlas.ChartOptions()
    if large_mesh_mode:
        chart_options.max_iterations = 1
        chart_options.max_cost = 3.0
    else:
        chart_options.max_iterations = max_iterations
    chart_options.normal_seam_weight = 0.5
    chart_options.texture_seam_weight = 1.0

    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.resolution = resolution
    pack_options.bilinear = True
    if large_mesh_mode:
        pack_options.rotate_charts = False

    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, indices, uvs = atlas[0]

    # Update the mesh
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def open3d_mesh_uv_wrap(mesh, gutter_size=2.0, max_stretch=0.06, resolution=1024):
    try:
        import open3d as o3d
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        o3d_mesh = o3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = o3d.core.Tensor(mesh.vertices)
        o3d_mesh.triangle.indices = o3d.core.Tensor(mesh.faces)

        core_count = os.cpu_count()
        print('Using Open3D for UV unwrapping with {} cores'.format(core_count))

        o3d_mesh.compute_uvatlas(
            size=resolution,
            parallel_partitions=4,
            gutter=gutter_size,
            max_stretch=max_stretch,
            nthreads=core_count
        )

        new_v = mesh.vertices[mesh.faces.reshape(-1)]
        new_f = np.arange(len(new_v)).reshape(-1, 3)
        new_uv = o3d_mesh.triangle.texture_uvs.numpy().reshape(-1, 2)

        mesh = trimesh.Trimesh(
            vertices=new_v,
            faces=new_f,
            process=False
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=new_uv.astype(np.float32),
        )
    except Exception as e:
        # Open3D might fail on mesh conditions so we will fallback to xatlas
        print('Open3D failed to unwrap mesh, falling back to xatlas. Error: ', e)
        return mesh_uv_wrap(mesh)

    return mesh


def bpy_unwrap_mesh(mesh):
    import bpy
    import bmesh

    # Store original vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Clear any existing mesh with the same name
    if "TempMesh" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes["TempMesh"])
    if "TempObject" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["TempObject"])

    # Create new mesh and object
    bpy_mesh = bpy.data.meshes.new(name="TempMesh")
    obj = bpy.data.objects.new(name="TempObject", object_data=bpy_mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)

    # Set as active object with proper context
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Create BMesh and populate
    bm = bmesh.new()
    vert_list = [bm.verts.new(tuple(v)) for v in vertices]
    bm.verts.ensure_lookup_table()

    # Add faces with error checking
    for f in faces:
        try:
            face_verts = [vert_list[i] for i in f]
            bm.faces.new(face_verts)
        except ValueError as e:
            print(f"Skipping invalid face: {e}")
            continue

    # Update mesh
    bm.to_mesh(bpy_mesh)
    bm.free()

    # Ensure UV layer exists
    if not bpy_mesh.uv_layers:
        bpy_mesh.uv_layers.new(name="UVMap")

    # Switch to edit mode and unwrap with proper context
    override = bpy.context.copy()
    override['active_object'] = obj
    override['object'] = obj
    override['edit_object'] = obj
    override['scene'] = bpy.context.scene

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        override,
        angle_limit=66.0,
        island_margin=0.03
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get UV data
    uv_layer = bpy_mesh.uv_layers.active
    if not uv_layer:
        print("Failed to create UV layer")
        return mesh

    # Extract UV coordinates
    uv_data = uv_layer.data
    uvs = np.zeros((len(faces), 3, 2), dtype=np.float32)

    for poly in bpy_mesh.polygons:
        for loop_idx, loop in enumerate(poly.loop_indices):
            uv = uv_data[loop].uv
            uvs[poly.index, loop_idx] = [uv.x, uv.y]

    # Create averaged UVs per vertex
    vertex_uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    counts = np.zeros(len(vertices), dtype=np.int32)

    for face_idx, face in enumerate(faces):
        for vert_idx, uv in zip(face, uvs[face_idx]):
            vertex_uvs[vert_idx] += uv
            counts[vert_idx] += 1

    # Avoid division by zero and compute average
    mask = counts > 0
    vertex_uvs[mask] /= counts[mask, None]

    # Create new TextureVisuals object
    mesh.visual = trimesh.visual.TextureVisuals(uv=vertex_uvs)

    # Clean up
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(bpy_mesh)

    return mesh
