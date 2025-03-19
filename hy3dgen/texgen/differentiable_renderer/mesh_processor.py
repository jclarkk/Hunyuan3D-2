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


def meshVerticeInpaint_smooth_optimized(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]
    num_faces = uv_idx.shape[0]

    flat_uv_idx = uv_idx.flatten()
    uv_coords = vtx_uv[flat_uv_idx]

    uv_v_all = np.rint(uv_coords[:, 0] * (texture_width - 1)).astype(np.int32)
    uv_u_all = np.rint((1.0 - uv_coords[:, 1]) * (texture_height - 1)).astype(np.int32)

    uv_v_all = uv_v_all.reshape(uv_idx.shape)
    uv_u_all = uv_u_all.reshape(uv_idx.shape)

    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = np.zeros((vtx_num, texture_channel), dtype=np.float32)

    src = pos_idx.reshape(-1)
    dst = np.empty_like(src)
    dst[0::3] = pos_idx[:, 1].reshape(-1)
    dst[1::3] = pos_idx[:, 2].reshape(-1)
    dst[2::3] = pos_idx[:, 0].reshape(-1)

    G = [[] for _ in range(vtx_num)]
    for s, d in zip(src, dst):
        G[s].append(d)

    uncolored_set = set()
    for i in range(num_faces):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            u = uv_u_all[i, k]
            v = uv_v_all[i, k]
            if mask[u, v] > 0:
                vtx_mask[vtx_idx] = 1.0
                vtx_color[vtx_idx] = texture[u, v]
            else:
                uncolored_set.add(vtx_idx)
    uncolored_vtxs = np.array(list(uncolored_set), dtype=np.int32)

    smooth_count = 2
    last_uncolored_vtx_count = -1
    while smooth_count > 0:
        new_uncolored = []
        for vtx_idx in uncolored_vtxs:
            pos0 = vtx_pos[vtx_idx]
            neighbors = G[vtx_idx]
            if len(neighbors) == 0:
                new_uncolored.append(vtx_idx)
                continue
            neighbor_positions = vtx_pos[neighbors]
            diffs = neighbor_positions - pos0
            dists = np.linalg.norm(diffs, axis=1)

            weights = 1.0 / np.maximum(dists, 1e-4)
            weights = weights ** 2

            neighbor_colored = (vtx_mask[neighbors] > 0)
            if np.any(neighbor_colored):
                valid_weights = weights[neighbor_colored]
                total_weight = valid_weights.sum()
                if total_weight > 0:
                    neighbor_colors = vtx_color[neighbors]
                    weighted_sum = np.sum(neighbor_colors[neighbor_colored] * valid_weights[:, None], axis=0)
                    vtx_color[vtx_idx] = weighted_sum / total_weight
                    vtx_mask[vtx_idx] = 1.0
                else:
                    new_uncolored.append(vtx_idx)
            else:
                new_uncolored.append(vtx_idx)
        new_uncolored = np.array(new_uncolored, dtype=np.int32)
        uncolored_count = new_uncolored.shape[0]

        if uncolored_count == last_uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_count
        uncolored_vtxs = new_uncolored

    new_texture = texture.copy()
    new_mask = mask.copy()
    for i in range(num_faces):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            if vtx_mask[vtx_idx] == 1.0:
                u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
                v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
                new_texture[u, v] = vtx_color[vtx_idx]
                new_mask[u, v] = 255
    return new_texture, new_mask


def meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = [np.zeros(texture_channel, dtype=np.float32) for _ in range(vtx_num)]
    uncolored_vtxs = []
    G = [[] for _ in range(vtx_num)]

    for i in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
            uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
            if mask[uv_u, uv_v] > 0:
                vtx_mask[vtx_idx] = 1.0
                vtx_color[vtx_idx] = texture[uv_u, uv_v]
            else:
                uncolored_vtxs.append(vtx_idx)
            G[pos_idx[i, k]].append(pos_idx[i, (k + 1) % 3])

    smooth_count = 2
    last_uncolored_vtx_count = 0
    while smooth_count > 0:
        uncolored_vtx_count = 0
        for vtx_idx in uncolored_vtxs:
            sum_color = np.zeros(texture_channel, dtype=np.float32)
            total_weight = 0.0
            vtx_0 = vtx_pos[vtx_idx]
            for connected_idx in G[vtx_idx]:
                if vtx_mask[connected_idx] > 0:
                    vtx1 = vtx_pos[connected_idx]
                    dist = np.sqrt(np.sum((vtx_0 - vtx1) ** 2))
                    dist_weight = 1.0 / max(dist, 1e-4)
                    dist_weight *= dist_weight
                    sum_color += vtx_color[connected_idx] * dist_weight
                    total_weight += dist_weight
            if total_weight > 0:
                vtx_color[vtx_idx] = sum_color / total_weight
                vtx_mask[vtx_idx] = 1.0
            else:
                uncolored_vtx_count += 1

        if last_uncolored_vtx_count == uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_vtx_count

    new_texture = texture.copy()
    new_mask = mask.copy()
    for face_idx in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[face_idx, k]
            vtx_idx = pos_idx[face_idx, k]
            if vtx_mask[vtx_idx] == 1.0:
                uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
                uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))
                new_texture[uv_u, uv_v] = vtx_color[vtx_idx]
                new_mask[uv_u, uv_v] = 255
    return new_texture, new_mask

def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    if method == "smooth":
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
    else:
        raise ValueError("Invalid method. Use 'smooth' or 'forward'.")