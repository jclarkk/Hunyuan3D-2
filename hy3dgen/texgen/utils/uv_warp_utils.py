# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh
import xatlas


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 100000:
        raise ValueError("The mesh has more than 100,000 faces, which is not supported.")

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def bpy_unwrap_mesh(vertices, faces):
    import bpy
    import bmesh
    import numpy as np
    # Create a new mesh and object
    mesh = bpy.data.meshes.new(name="TempMesh")
    obj = bpy.data.objects.new(name="TempObject", object_data=mesh)

    # Link the object to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Create a BMesh object and add geometry
    bm = bmesh.new()

    # Add vertices
    vert_list = [bm.verts.new(tuple(v)) for v in vertices]
    bm.verts.ensure_lookup_table()

    # Add faces
    for f in faces:
        bm.faces.new([vert_list[i] for i in f])

    # Write the BMesh to the mesh data
    bm.to_mesh(mesh)
    bm.free()

    # Enter edit mode to unwrap
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Extract UV coordinates
    uv_layer = mesh.uv_layers.active.data
    uvs = np.array([uv.uv for uv in uv_layer], dtype=np.float32)

    # Cleanup: Remove the temporary object from the scene
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)

    return uvs
