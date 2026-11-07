#!/usr/bin/env python3
"""
Encode variable-size 3D point clouds with a pretrained PointNet++ (SSG, ModelNet40),
then compare them with cosine similarity.

Setup (clone into the project folder, next to this file):
    git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    # Pretrained weights ship in the repo:
    # Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth

Run via `python main.py classify` (reads the detected-object CSV, writes similarities.csv;
settings live in classification_config.py).

Sampling strategies (how a cloud is reduced / padded to n_points):
    "random"  : random subset (non-deterministic; padding = random repeats)
    "uniform" : evenly spaced indices through the point list (deterministic, fast)
    "fps"     : farthest point sampling on GPU (deterministic, best coverage, slower)
    "voxel"   : voxel-grid downsampling, one point per voxel (deterministic, even density)
Deterministic modes pad small clouds by cycling through their points in order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from classification_config import N_POINTS, SAMPLING, SIMILARITIES_PATH, TARGETS, THRESHOLD
from detection_config import OBJECT_POINTS_PATH

REPO = Path(__file__).resolve().parent / "Pointnet_Pointnet2_pytorch"
CKPT = REPO / "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
if not REPO.is_dir():
    raise FileNotFoundError(
        f"PointNet++ repo not found: {REPO}\n"
        f'Clone it with: git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch "{REPO}"'
    )
sys.path.append(str(REPO))
sys.path.append(str(REPO / "models"))
from pointnet2_cls_ssg import get_model  # noqa: E402

SAMPLING_MODES = ("random", "uniform", "fps", "voxel")


# ------------------------------------------------------------ sampling methods
def sample_uniform(pc: torch.Tensor, n: int) -> torch.Tensor:
    """n evenly spaced indices through the cloud (N >= n)."""
    return torch.linspace(0, pc.shape[0] - 1, n, device=pc.device).round().long()


def sample_fps(pc: torch.Tensor, n: int, pre_cap: int = 8192) -> torch.Tensor:
    """
    Farthest point sampling on the GPU, deterministic start (point farthest from centroid).
    Clouds larger than pre_cap are first thinned by uniform stride so the n-step
    loop stays cheap. Returns indices into pc.
    """
    N = pc.shape[0]
    base = sample_uniform(pc, pre_cap) if N > pre_cap else torch.arange(N, device=pc.device)
    p = pc[base]
    idx = torch.empty(n, dtype=torch.long, device=pc.device)
    dist = torch.full((p.shape[0],), float("inf"), device=pc.device)
    far = ((p - p.mean(0)) ** 2).sum(1).argmax()
    for i in range(n):
        idx[i] = far
        dist = torch.minimum(dist, ((p - p[far]) ** 2).sum(1))
        far = dist.argmax()
    return base[idx]


def sample_voxel(pc: torch.Tensor, n: int, iters: int = 14) -> torch.Tensor:
    """
    Voxel-grid downsampling: bisect the voxel size to the largest one that still
    yields >= n occupied voxels, keep one point per voxel, then thin evenly to n.
    """
    N = pc.shape[0]
    mins = pc.min(0).values
    lo, hi = 1e-4, 2.0
    best = None
    for _ in range(iters):
        v = (lo * hi) ** 0.5
        keys = torch.floor((pc - mins) / v).long()
        _, inv = torch.unique(keys, dim=0, return_inverse=True)
        cnt = int(inv.max()) + 1
        if cnt >= n:
            best, lo = (inv, cnt), v
        else:
            hi = v
    if best is None:                      # too many duplicate points
        return sample_uniform(pc, n)
    inv, cnt = best
    first = torch.full((cnt,), N, dtype=torch.long, device=pc.device)
    first = first.scatter_reduce(0, inv, torch.arange(N, device=pc.device), "amin")
    return first[sample_uniform(first.unsqueeze(1), n)] if cnt > n else first


def normalize(pc: torch.Tensor) -> torch.Tensor:
    """Center at origin and scale into the unit sphere (as in ModelNet40)."""
    pc = pc - pc.mean(0, keepdim=True)
    return pc / (pc.norm(dim=1).max() + 1e-9)


def to_fixed_size(pc: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    """Make any (N, 3) cloud exactly (n, 3) using the chosen sampling mode."""
    N, dev = pc.shape[0], pc.device
    if N == n:
        return pc
    if N < n:                             # pad
        if mode == "random":
            idx = torch.cat([torch.arange(N, device=dev), torch.randint(0, N, (n - N,), device=dev)])
        else:
            idx = torch.arange(n, device=dev) % N     # deterministic cycling
        return pc[idx]
    if mode == "random":                  # reduce
        idx = torch.randperm(N, device=dev)[:n]
    elif mode == "uniform":
        idx = sample_uniform(pc, n)
    elif mode == "fps":
        idx = sample_fps(pc, n)
    elif mode == "voxel":
        idx = sample_voxel(pc, n)
    else:
        raise ValueError(f"sampling must be one of {SAMPLING_MODES}")
    return pc[idx]


# ---------------------------------------------------------------------- encoder
class PointNet2Encoder:
    def __init__(self, ckpt=CKPT, n_points=4000, sampling="uniform", device=None):
        """n_points must be >= 512 (the SA layers sample 512, then 128 centroids)."""
        assert sampling in SAMPLING_MODES, f"sampling must be one of {SAMPLING_MODES}"
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"PointNet++ checkpoint not found: {ckpt}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_points, self.sampling = n_points, sampling

        self.model = get_model(num_class=40, normal_channel=False)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)  # trusted repo checkpoint
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device).eval()
        print(f"PointNet++ running on: {self.device} | sampling: {sampling}")

    def _prep(self, c) -> torch.Tensor:
        pc = torch.as_tensor(c, dtype=torch.float32).to(self.device, non_blocking=True)
        return to_fixed_size(normalize(pc), self.n_points, self.sampling)

    @torch.inference_mode()
    def encode(self, clouds, batch_size=128, verbose=True) -> np.ndarray:
        """
        clouds: list of (N_i, 3) arrays with differing N_i (a single array also works).
        returns: (len(clouds), 1024) global feature vectors.
        """
        if isinstance(clouds, np.ndarray) and clouds.ndim == 2:
            clouds = [clouds]

        feats = []
        for i in range(0, len(clouds), batch_size):
            x = torch.stack([self._prep(c) for c in clouds[i:i + batch_size]])  # (B, n, 3)
            _, l3_points = self.model(x.transpose(1, 2))                        # (B, 1024, 1)
            feats.append(l3_points.flatten(1).float().cpu())
            if verbose:
                print(f"encoded {min(i + batch_size, len(clouds))}/{len(clouds)}", end="\r")
        if verbose:
            print()
        return torch.cat(feats).numpy()


# ------------------------------------------------------------------ similarity
def cosine_similarity_matrix(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """(Q, D) x (T, D) -> (Q, T) cosine similarities."""
    q = F.normalize(torch.from_numpy(queries).float(), dim=1)
    t = F.normalize(torch.from_numpy(targets).float(), dim=1)
    return (q @ t.T).numpy()


def class_prototypes(enc, class_clouds, batch_size=128) -> np.ndarray:
    """
    class_clouds: list of length T; each item is one (N, 3) cloud or a list of clouds
    for that class (their features are averaged). returns: (T, 1024)
    """
    protos = []
    for item in class_clouds:
        group = [item] if isinstance(item, np.ndarray) else list(item)
        protos.append(enc.encode(group, batch_size, verbose=False).mean(0))
    return np.stack(protos)


NOT_IN_TARGETS = "not in targets"


def compare(enc, class_clouds, query_clouds, class_names, threshold=0.95, batch_size=128):
    """
    Returns:
        sims   : (Q, T) cosine similarity of every query to every target class
        best   : (Q,)   index of the most similar class per query
        score  : (Q,)   its similarity
        labels : list of Q strings: the best class name if score >= threshold,
                 otherwise "not in targets"
    """
    targets = class_prototypes(enc, class_clouds, batch_size)   # (T, 1024)
    queries = enc.encode(query_clouds, batch_size)              # (Q, 1024)
    sims = cosine_similarity_matrix(queries, targets)
    best = sims.argmax(1)
    score = sims[np.arange(len(sims)), best]
    labels = [class_names[b] if sc >= threshold else NOT_IN_TARGETS
              for b, sc in zip(best, score)]
    return sims, best, score, labels


# ---------------------------------------------------------------------- CSV I/O
def load_clusters_csv(path, skip_noise=True, min_points=10):
    """
    Read a CSV with a header row and columns: cluster_id, x, y, z (one point per row).
    Returns (ids, clouds): ids is a list of cluster ids, clouds a list of (N_i, 3) float32 arrays.
    - skip_noise: drop negative ids (e.g. -1 = noise in DBSCAN)
    - min_points: drop clusters smaller than this
    """
    df = pd.read_csv(path)                                   # first row = header
    df = df.iloc[:, :4].dropna()
    df.columns = ["cid", "x", "y", "z"]
    if skip_noise:
        df = df[df["cid"] >= 0]

    ids, clouds = [], []
    for cid, g in df.groupby("cid", sort=True):
        if len(g) < min_points:
            continue
        ids.append(cid)
        clouds.append(g[["x", "y", "z"]].to_numpy(dtype=np.float32))
    if not ids:
        raise ValueError(f"{path} contains no clusters with at least {min_points} points")
    print(f"loaded {len(ids)} clusters, {len(df)} points "
          f"(sizes {min(map(len, clouds))}..{max(map(len, clouds))})")
    return ids, clouds


def split_targets_queries(ids, clouds, targets=None, n_targets=10):
    """
    Pick the target classes from the clusters; every other cluster is a query.
    - targets: dict {class_name: cluster_id}, or {class_name: [id, id, ...]} to average
      several clusters into one class, e.g. {"Valve": 4, "Pump": [10, 11]}.
      If None, the first n_targets clusters are used, named by their id.
    """
    by_id = dict(zip(ids, clouds))
    if targets is None:
        targets = {str(i): i for i in ids[:n_targets]}

    target_names, target_clouds, used = [], [], set()
    for name, t in targets.items():
        group = list(t) if isinstance(t, (list, tuple)) else [t]
        missing = [g for g in group if g not in by_id]
        if missing:
            raise KeyError(f"target '{name}': cluster id(s) {missing} not found "
                           f"(or removed by skip_noise/min_points)")
        target_names.append(name)
        target_clouds.append(by_id[group[0]] if len(group) == 1 else [by_id[g] for g in group])
        used.update(group)

    query_ids = [i for i in ids if i not in used]
    query_clouds = [by_id[i] for i in query_ids]
    print(f"{len(target_names)} target classes, {len(query_ids)} queries")
    return target_names, target_clouds, query_ids, query_clouds


def main() -> None:
    """Match every detected object to the target classes and write similarities.csv."""
    if not OBJECT_POINTS_PATH.is_file():
        raise FileNotFoundError(
            f"Detected-object CSV not found: {OBJECT_POINTS_PATH}. "
            "Run `python main.py detect` first."
        )

    ids, clouds = load_clusters_csv(OBJECT_POINTS_PATH)
    target_names, target_clouds, query_ids, query_clouds = split_targets_queries(
        ids, clouds, targets=TARGETS)

    enc = PointNet2Encoder(n_points=N_POINTS, sampling=SAMPLING)
    sims, best, score, labels = compare(enc, target_clouds, query_clouds, target_names,
                                        threshold=THRESHOLD)

    n_match = sum(l != NOT_IN_TARGETS for l in labels)
    print(f"{n_match}/{len(labels)} queries matched a target (cosine >= {THRESHOLD})")
    for i in range(len(query_ids)):
        print(f"query {query_ids[i]}: {labels[i]}  "
              f"(best was {target_names[best[i]]}, cos={score[i]:.3f})")

    # Similarity of each target to every target (so targets appear in the table too)
    t_protos = class_prototypes(enc, target_clouds)
    t_sims = cosine_similarity_matrix(t_protos, t_protos)           # (T, T)
    t_best = t_sims.argmax(1)
    t_ids = ["+".join(map(str, v)) if isinstance(v, (list, tuple)) else v for v in TARGETS.values()]

    cols = [f"sim_target_{n}" for n in target_names]
    rows_t = pd.DataFrame(t_sims, columns=cols)
    rows_t.insert(0, "query_id", t_ids)
    rows_t.insert(1, "label", [f"Target {n}" for n in target_names])
    rows_t.insert(2, "best_target", [target_names[b] for b in t_best])
    rows_t.insert(3, "best_score", t_sims[np.arange(len(t_sims)), t_best])

    rows_q = pd.DataFrame(sims, columns=cols)
    rows_q.insert(0, "query_id", query_ids)
    rows_q.insert(1, "label", labels)
    rows_q.insert(2, "best_target", [target_names[b] for b in best])
    rows_q.insert(3, "best_score", score)

    pd.concat([rows_t, rows_q], ignore_index=True).to_csv(SIMILARITIES_PATH, index=False)
    print(f"Saved similarities to {SIMILARITIES_PATH}")


if __name__ == "__main__":
    main()
