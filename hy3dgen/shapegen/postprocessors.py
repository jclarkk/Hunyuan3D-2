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

import os
import tempfile
from typing import Union

import numpy as np
import pymeshlab
import torch
import pynanoinstantmeshes as PyNIM
import trimesh

from .models.autoencoders import Latent2MeshOutput
from .utils import synchronize_timer


def remesh_with_meshlib(mesh: trimesh.Trimesh):
    import meshlib.mrmeshpy as mrmeshpy
    import meshlib.mrmeshnumpy as mrmeshnumpy
    import numpy as np
    import random

    # Load mesh
    mesh = mrmeshnumpy.meshFromFacesVerts(mesh.faces, mesh.vertices)
    topo = mesh.topology
    original_faces = topo.numValidFaces()

    # Calculate average edge length
    total_length = 0.0
    edge_count = 0
    valid_edges = topo.findNotLoneUndirectedEdges()

    # Convert iterator to list and sample from it
    edge_list = []
    for edge_id in valid_edges:
        edge_list.append(edge_id)

    # Sample up to 1000 edges
    sample_size = min(1000, len(edge_list))
    sampled_edges = random.sample(edge_list, sample_size)

    for edge_id in sampled_edges:
        org_id = topo.org(edge_id)
        dest_id = topo.dest(edge_id)

        # Get vertex coordinates using subscript operator
        try:
            org_point = mesh.points[org_id]
            dest_point = mesh.points[dest_id]
        except:
            org_point = mesh.points.autoResizeAt(org_id)
            dest_point = mesh.points.autoResizeAt(dest_id)

        # Calculate edge length
        edge_vec = np.array([
            dest_point.x - org_point.x,
            dest_point.y - org_point.y,
            dest_point.z - org_point.z
        ])
        edge_length = np.linalg.norm(edge_vec)
        total_length += edge_length
        edge_count += 1

    avg_edge_len = total_length / edge_count if edge_count > 0 else 0.0
    print(f"Average edge length: {avg_edge_len}")

    # Create remesh settings for decimation
    settings = mrmeshpy.RemeshSettings()

    # Set remeshing parameters for better curvature preservation
    settings.useCurvature = True  # Enable curvature-based adaptation
    settings.maxEdgeSplits = 0  # Disable splitting edges (only reduce)

    # Use a more conservative target edge length - less aggressive
    target_ratio = min(3.0, (original_faces / 100000.0) ** 0.5)
    settings.targetEdgeLen = avg_edge_len * target_ratio

    # More conservative angle change to preserve features
    settings.maxAngleChangeAfterFlip = 0.3  # Lower value preserves shape better

    # Limit boundary movement
    settings.maxBdShift = avg_edge_len * 0.1

    settings.finalRelaxNoShrinkage = True
    settings.projectOnOriginalMesh = True

    # Use more relaxation iterations for better shape preservation
    settings.finalRelaxIters = 3

    # Run remeshing
    print(f"Starting controlled decimation with target edge length ratio: {target_ratio:.2f}...")
    mrmeshpy.remesh(mesh, settings)

    # Report results
    new_face_count = topo.numValidFaces()
    reduction_percent = ((original_faces - new_face_count) / original_faces) * 100
    print(f"Decimation complete: {new_face_count} faces (reduced by {reduction_percent:.1f}%)")

    # Convert back to trimesh
    out_verts = mrmeshnumpy.getNumpyVerts(mesh)
    out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

    return trimesh.Trimesh(out_verts, out_faces)


def clean_mesh_with_meshlib(mesh: trimesh.Trimesh):
    import meshlib.mrmeshpy as mrmeshpy
    import meshlib.mrmeshnumpy as mrmeshnumpy

    # Load mesh
    mesh = mrmeshnumpy.meshFromFacesVerts(mesh.faces, mesh.vertices)

    # Find single edge for each hole in mesh
    hole_edges = mesh.topology.findHoleRepresentiveEdges()

    if len(hole_edges) > 0:
        print('Found holes in mesh, will attempt to fix.')

    for e in hole_edges:
        #  Setup filling parameters
        params = mrmeshpy.FillHoleParams()
        params.metric = mrmeshpy.getUniversalMetric(mesh)
        #  Fill hole represented by `e`
        mrmeshpy.fillHole(mesh, e, params)

    out_verts = mrmeshnumpy.getNumpyVerts(mesh)
    out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

    return trimesh.Trimesh(out_verts, out_faces)


def reduce_face_with_meshlib(mesh: trimesh.Trimesh, max_facenum: int = 100000):
    current_face_count = len(mesh.faces)
    if current_face_count <= max_facenum:
        return mesh

    import meshlib.mrmeshpy as mrmeshpy
    import meshlib.mrmeshnumpy as mrmeshnumpy
    import multiprocessing

    # Load mesh
    mesh = mrmeshnumpy.meshFromFacesVerts(mesh.faces, mesh.vertices)

    faces_to_delete = current_face_count - max_facenum
    #  Setup simplification parameters
    mesh.packOptimally()
    settings = mrmeshpy.DecimateSettings()
    settings.maxDeletedFaces = faces_to_delete
    settings.subdivideParts = multiprocessing.cpu_count()
    settings.maxError = 0.05
    settings.packMesh = True

    print(f'Decimating mesh... Deleting {faces_to_delete} faces')
    mrmeshpy.decimateMesh(mesh, settings)
    print(f'Decimation done. Resulting mesh has {mesh.topology.faceSize()} faces')

    out_verts = mrmeshnumpy.getNumpyVerts(mesh)
    out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

    return trimesh.Trimesh(out_verts, out_faces)


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

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
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
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
    @synchronize_timer('FaceReducer')
    def __call__(
            self,
            mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
            max_facenum: int = 100000,
            remesh_method: str = None,
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        target_vertex_count = int(max_facenum / 8)

        print(f"Reducing face count to {max_facenum}...")
        if remesh_method is not None and remesh_method == "im":
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
        elif remesh_method is not None and remesh_method == "bpt":
            from .bpt.pipeline import BPTPipeline
            pipeline = BPTPipeline.from_pretrained()
            mesh = pipeline(mesh)
        elif remesh_method is not None and remesh_method == "deepmesh":
            from .DeepMesh.pipeline import DeepMeshPipeline
            pipeline = DeepMeshPipeline.from_pretrained()
            mesh = pipeline(mesh)
        elif len(mesh.faces) > max_facenum:
            mesh = reduce_face_with_meshlib(mesh, max_facenum)

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


class MeshlibCleaner:
    @synchronize_timer('MeshlibCleaner')
    def __call__(
            self,
            mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        mesh = clean_mesh_with_meshlib(mesh)
        return mesh


class FloaterRemover:
    @synchronize_timer('FloaterRemover')
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

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file.name)

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable

    @synchronize_timer('MeshSimplifier')
    def __call__(
            self,
            mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                mesh.export(temp_input.name)
                os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                ms = trimesh.load(temp_output.name, process=False)
                if isinstance(ms, trimesh.Scene):
                    combined_mesh = trimesh.Trimesh()
                    for geom in ms.geometry.values():
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                    ms = combined_mesh
                ms = mesh_normalize(ms)
                return ms
