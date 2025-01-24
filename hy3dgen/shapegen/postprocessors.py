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

import numpy as np
import os
import tempfile
from typing import Union

import pynanoinstantmeshes as PyNIM
import pymeshlab
import trimesh

from .models.vae import Latent2MeshOutput


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=True)
    temp_file.close()
    temp_file_name = temp_file.name

    mesh.save_current_mesh(temp_file_name)
    mesh = trimesh.load(temp_file_name)
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=True)
    temp_file.close()
    temp_file_name = temp_file.name

    if isinstance(mesh, trimesh.scene.Scene):
        for idx, obj in enumerate(mesh.geometry.values()):
            if idx == 0:
                temp_mesh = obj
            else:
                temp_mesh = temp_mesh + obj
        mesh = temp_mesh
    mesh.export(temp_file_name)
    mesh = pymeshlab.MeshSet()
    mesh.load_new_mesh(temp_file_name)
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    return mesh


def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


class FaceReducer:
    def __call__(
            self,
            mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
            max_facenum: int = 50000,
            im_remesh: bool = False
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        target_vertex_count = int(max_facenum / 8)

        print(f"Reducing face count to {max_facenum}...")
        if im_remesh:
            vertices, faces = PyNIM.remesh(
                np.array(mesh.vertices, dtype=np.float32),
                np.array(mesh.faces, dtype=np.uint32),
                target_vertex_count,
                align_to_boundaries=True,
                smooth_iter=8
            )
            vertices = vertices.astype(np.float32)
            faces = self.quads_to_triangles(faces)
            mesh = trimesh.Trimesh(vertices, faces)
            mesh = trimesh.smoothing.filter_laplacian(mesh)

            if len(mesh.faces) > max_facenum:
                ms = import_mesh(mesh)
                ms = reduce_face(ms, max_facenum=max_facenum)
                current_mesh = ms.current_mesh()
                mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

        print(f"Resulting mesh has {len(mesh.faces)} faces")

        return mesh

    @staticmethod
    def quads_to_triangles(quads):
        triangles = []

        for quad in quads:
            if len(quad) != 4:
                raise ValueError("Each quad must have exactly 4 vertices.")

            triangles.append([quad[0], quad[1], quad[2]])
            triangles.append([quad[0], quad[2], quad[3]])

        return np.array(triangles)


class FloaterRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=True)
        temp_file.close()
        temp_file_name = temp_file.name

        ms.save_current_mesh(temp_file_name)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_file_name)
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

        mesh = export_mesh(mesh, ms)
        return mesh
