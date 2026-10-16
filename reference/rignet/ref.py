"""RigNet's reference outputs, network by network, for `nextsycl model3d check --model rignet` (model3d/rignet/src/check.rs).

Runs github.com/zhan-xu/RigNet's own models (models/*.py) and numpy steps (utils/cluster_utils.py, utils/mst_utils.py,
geometric_proc/*: pts2line, the trimesh ray visibility) in float32 on the CPU, on the geometry nextsycl made (its
`inputs` stage: the decimated, normalized mesh, its edges, geodesic edges, surface geodesics and voxels), so each
network and step is compared on the same input. Stand-ins (stand_ins/): torch_geometric / torch_scatter in pure torch
(fps starting at the first point, as the port), an empty open3d. The checkpoints come out of trained_models.zip.

    python ref.py --repo SRC/RigNet --zip ~/models/rignet/trained_models.zip --dir ~/nsref/rig/rignet

Reads pos, faces, tpl, geo, geodesic, vox, vox_meta (.npy) from --dir and writes there:
    disp.npy [n, 3], attn.npy [n], bandwidth.npy [1]    JointNet (tanh displacement, sigmoid attention)
    joints.npy [J, 3]                                    quick_start.predict_joints's joints (threshold 1e-5)
    pair_input.npy [P, 8], root_logits.npy [J], root_id.npy [1], bone_logits.npy [P], parent.npy [J]
    bones.npy [B, 6], geo_dist.npy [n, B]                calc_geodesic_matrix (every vertex, no subsampling)
    skin_nn.npy, skin_mask.npy [n, 5], skin_input.npy [n, 43], skin_logits.npy [n, 5], skin_full.npy [n, B]
"""
import argparse
import io
import itertools as it
import os
import sys
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def save(d, name, x):
    np.save(os.path.join(d, name + ".npy"), np.ascontiguousarray(np.asarray(x, dtype=np.float32)))


def load(d, name):
    return np.load(os.path.join(d, name + ".npy"))


def post_filter(skin_weights, topology_edge, num_ring=1):
    """run_skinning.post_filter, as written there (run_skinning.py imports the training stack)"""
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
    return skin_weights_new


class Vox:
    def __init__(self, data, translate, scale):
        self.data, self.translate, self.scale = data, translate, scale
        self.dims = list(data.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count()))
    sys.path.insert(0, os.path.join(HERE, "stand_ins"))
    sys.path.insert(0, a.repo)
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops
    from models.GCN import JOINTNET_MASKNET_MEANSHIFT as JOINTNET
    from models.ROOT_GCN import ROOTNET
    from models.PairCls_GCN import PairCls as BONENET
    from models.SKINNING import SKINNET
    from utils.cluster_utils import meanshift_cluster, nms_meanshift
    from utils.mst_utils import increase_cost_for_outside_bone, primMST_symmetry, loadSkel_recur, inside_check, flip, sample_on_bone
    from utils.rig_parser import Info
    from utils.tree_utils import TreeNode
    from geometric_proc.common_ops import get_bones
    from geometric_proc.compute_volumetric_geodesic import pts2line, calc_pts2bone_visible_mat
    import trimesh

    z = zipfile.ZipFile(a.zip)

    def ckpt(name):
        return torch.load(io.BytesIO(z.read("checkpoints/%s/model_best.pth.tar" % name)), map_location="cpu", weights_only=False)["state_dict"]

    d = a.dir
    pos = torch.from_numpy(load(d, "pos")).float()
    faces = load(d, "faces").astype(np.int64)
    n = pos.shape[0]
    tpl = torch.from_numpy(load(d, "tpl").astype(np.int64)).T
    tpl, _ = add_self_loops(tpl, num_nodes=n)
    geo = torch.from_numpy(load(d, "geo").astype(np.int64)).T
    geo, _ = add_self_loops(geo, num_nodes=n)
    meta = load(d, "vox_meta")
    vox = Vox(load(d, "vox") > 0.5, meta[0:3].astype(np.float64), float(meta[3]))
    surface_geodesic = load(d, "geodesic").astype(np.float64)
    surface_geodesic += 1.0 * np.eye(n)  # as gen_dataset.get_geo_edges leaves it (in place)
    data = Data(x=None, pos=pos, tpl_edge_index=tpl, geo_edge_index=geo, batch=torch.zeros(n, dtype=torch.long))

    with torch.no_grad():
        jn = JOINTNET()
        jn.load_state_dict(ckpt("gcn_meanshift"))
        jn.eval()
        disp, _, attn, bandwidth = jn(data)
        save(d, "disp", disp)
        save(d, "attn", attn[:, 0])
        save(d, "bandwidth", bandwidth.reshape(1))

        # quick_start.predict_joints (threshold 1e-5, the learned bandwidth)
        threshold = 1e-5
        y = (disp + data.pos).numpy()
        at = attn.numpy()
        y, inside = inside_check(y, vox)
        at = at[inside, :]
        y = y[at.squeeze() > 1e-3]
        at = at[at.squeeze() > 1e-3]
        y = np.concatenate((y, y * np.array([[-1, 1, 1]])), axis=0)
        at = np.tile(at, (2, 1))
        bw = bandwidth.item()
        y = meanshift_cluster(y, bw, at, max_iter=40)
        yd = np.sum(((y[np.newaxis, ...] - y[:, np.newaxis, :]) ** 2), axis=2)
        dens = np.maximum(bw ** 2 - yd, np.zeros(yd.shape)).sum(axis=0)
        keep = dens / np.sum(dens) > threshold
        y, dens = y[keep], dens[keep]
        joints = nms_meanshift(y, dens, bw)
        joints, _ = flip(joints)
        save(d, "joints", joints)
        print("joints", joints.shape[0])

        pairs = list(it.combinations(range(joints.shape[0]), 2))
        attr = []
        for pr in pairs:
            dist = np.linalg.norm(joints[pr[0]] - joints[pr[1]])
            s = sample_on_bone(joints[pr[0]], joints[pr[1]])
            si, _ = inside_check(s, vox)
            attr.append(np.array([dist, len(si) / (len(s) + 1e-10), 1]))
        pairs_t = torch.from_numpy(np.array(pairs)).float()
        attr_t = torch.from_numpy(np.array(attr)).float()
        jt = torch.from_numpy(joints).float()
        data.joints = jt
        data.pairs = pairs_t
        data.pair_attr = attr_t
        data.joints_batch = torch.zeros(len(jt), dtype=torch.long)
        data.pairs_batch = torch.zeros(len(pairs_t), dtype=torch.long)
        save(d, "pair_input", torch.cat((jt[pairs_t[:, 0].long()], jt[pairs_t[:, 1].long()], attr_t[:, :-1]), dim=1))

        rn = ROOTNET()
        rn.load_state_dict(ckpt("rootnet"))
        rn.eval()
        root_logits, _ = rn(data, shuffle=False)
        save(d, "root_logits", root_logits[:, 0])
        root_id = int(np.argmax(torch.sigmoid(root_logits).numpy()))
        save(d, "root_id", [root_id])

        bn = BONENET()
        bn.load_state_dict(ckpt("bonenet"))
        bn.eval()
        bone_logits, _ = bn(data, permute_joints=False)
        save(d, "bone_logits", bone_logits[:, 0])
        prob = torch.sigmoid(bone_logits).numpy()
        pm = np.zeros((len(jt), len(jt)))
        pi = pairs_t.long().numpy()
        pm[pi[:, 0], pi[:, 1]] = prob.squeeze()
        pm = pm + pm.T
        cost = increase_cost_for_outside_bone(-np.log(pm + 1e-10), joints, vox)
        parent, _, root_id = primMST_symmetry(cost, root_id, joints)
        save(d, "parent", [p for p in parent])

        skel = Info()
        for i in range(len(parent)):
            if parent[i] == -1:
                skel.root = TreeNode("root", tuple(joints[i]))
                break
        loadSkel_recur(skel.root, i, None, joints, parent)
        bones, _, bone_isleaf = get_bones(skel)
        save(d, "bones", bones)

        # quick_start.calc_geodesic_matrix without subsampling
        mesh = trimesh.Trimesh(vertices=pos.numpy().astype(np.float64), faces=faces, process=False)
        mesh_v = pos.numpy().astype(np.float64)
        origins, ends, pts_bone_dist = pts2line(mesh_v, bones)
        vis = calc_pts2bone_visible_mat(mesh, origins, ends).reshape(len(bones), n).transpose()
        dist = pts_bone_dist.reshape(len(bones), n).transpose()
        save(d, "vis_raw", vis)
        for b in range(vis.shape[1]):
            vp = np.argwhere(vis[:, b] == 1).squeeze(1)
            if len(vp) == 0:
                continue
            th = np.percentile(dist[vp, b], 15)
            vis[dist[:, b] > 1.3 * th, b] = False
        vm = np.zeros(vis.shape)
        vm[np.where(vis == 1)] = dist[np.where(vis == 1)]
        for c in range(vm.shape[1]):
            un = np.argwhere(vis[:, c] == 0).squeeze(1)
            vp = np.argwhere(vis[:, c] == 1).squeeze(1)
            if len(vp) == 0:
                vm[:, c] = dist[:, c]
                continue
            for r in un:
                d1 = np.min(surface_geodesic[r, vp])
                nnv = vp[np.argmin(surface_geodesic[r, vp])]
                vm[r, c] = 8.0 + dist[r, c] if np.isinf(d1) else d1 + vm[nnv, c]
        save(d, "geo_dist", vm)

        # quick_start.predict_skinning
        samples, nn, mask = [], [], []
        for v in range(n):
            g = vm[v]
            order = np.argsort(g)
            s, t, m = [], [], []
            for i in range(5):
                if i >= len(bones):
                    s += bones[order[0]].tolist() + [1.0 / (g[order[0]] + 1e-10), bone_isleaf[order[0]]]
                    t.append(0)
                    m.append(0)
                else:
                    s += bones[order[i]].tolist() + [1.0 / (g[order[i]] + 1e-10), bone_isleaf[order[i]]]
                    t.append(order[i])
                    m.append(1)
            samples.append(s)
            nn.append(t)
            mask.append(m)
        save(d, "skin_nn", nn)
        save(d, "skin_mask", mask)
        skin_input = torch.from_numpy(np.array(samples)).float()
        save(d, "skin_input", torch.cat([data.pos, skin_input], dim=1))
        data.skin_input = skin_input
        sn = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
        sn.load_state_dict(ckpt("skinnet"))
        sn.eval()
        logits = sn(data)
        save(d, "skin_logits", logits)
        sp = torch.softmax(logits, dim=1).numpy() * np.array(mask)
        full = np.zeros((n, len(bones)))
        for v in range(n):
            for k in range(5):
                full[v, nn[v][k]] = sp[v, k]
        full = post_filter(full, data.tpl_edge_index.numpy(), num_ring=1)
        full[full < np.max(full, axis=1, keepdims=True) * 0.35] = 0.0
        full = full / (full.sum(axis=1, keepdims=True) + 1e-10)
        save(d, "skin_full", full)
        print("bones", len(bones), "done")


if __name__ == "__main__":
    main()
