"""
PartSDF chair augmentation → Scene-Diffuser SDF format.

Based on jiin_augment_chairs.py. Saves each augmented chair as:
  object_X/
  ├── object_X.ply          # mesh
  ├── object_X.obj          # mesh
  ├── object_X_sdf.npy      # (D, D, D) float32  positive=outside, negative=inside
  ├── object_X_sdf_min.npy  # (3,) float32  grid bounding box min (meters)
  └── object_X_sdf_max.npy  # (3,) float32  grid bounding box max (meters)

SDF convention:
  positive = outside (safe)
  negative = inside  (penetration)
  Unit: meters
  Default grid: 64×64×64
  Bounding box: mesh bounds + ~0.1m margin

Usage:
  conda activate syncTweedies
  cd /home/user/bodyawarechair/PartSDF
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chairs_sdf.py --device cuda:0
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chairs_sdf.py --device cuda:0 --sdf-resolution 32  # faster
"""

import os
import sys
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.mesh import create_mesh
from src.model import get_model, get_part_latents, get_part_poses
from src.primitives import standardize_quaternion, slerp_quaternion
from src.utils import set_seed


# =========================================================================
# Part indices
# =========================================================================
BACKREST = 0
ARM_LEFT = 1
ARM_RIGHT = 2
LEG_FR = 3
LEG_BR = 4
LEG_FL = 5
LEG_BL = 6
SEAT = 7
LEGS = [LEG_FR, LEG_BR, LEG_FL, LEG_BL]
ARMS = [ARM_LEFT, ARM_RIGHT]


# =========================================================================
# Model loading
# =========================================================================
def load_model(expdir, device):
    specs = ws.load_specs(expdir)
    n_parts = specs["Parts"]["NumParts"]
    part_dim = specs["Parts"]["LatentDim"]
    model = get_model(
        specs["Network"], **specs.get("NetworkSpecs", {}),
        latent_dim=specs["LatentDim"], n_parts=n_parts, part_dim=part_dim,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    epoch = ws.load_history(expdir)["epoch"]
    ws.load_model(expdir, model, epoch)

    latent_ckpt = torch.load(
        os.path.join(expdir, "latent", f"latents_{epoch}.pth"), map_location="cpu"
    )
    n_train = latent_ckpt["weight"].shape[0]
    latents = get_part_latents(n_train, n_parts, part_dim,
                               specs.get("LatentBound", None), device=device)
    ws.load_latents(expdir, latents, epoch)
    poses = get_part_poses(n_train, n_parts, freeze=True, device=device, fill_nans=True)
    ws.load_poses(expdir, poses)

    print(f"Loaded: {n_train} chairs, {n_parts} parts, epoch={epoch}")
    return model, latents, poses, specs


def get_shape(latents, poses, idx, device):
    latent = latents(torch.tensor([idx]).to(device))
    R, t, s = poses(torch.tensor([idx]).to(device))
    return latent, R, t, s


# =========================================================================
# Watertight conversion
# =========================================================================
def make_closed_watertight(mesh, pitch=0.02, keep_largest=True):
    if mesh.is_empty:
        return mesh
    if keep_largest:
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            mesh = max(comps, key=lambda m: m.area)
    voxel = mesh.voxelized(pitch)
    closed = voxel.marching_cubes
    closed.remove_unreferenced_vertices()
    closed.fix_normals()
    if keep_largest:
        comps = closed.split(only_watertight=False)
        if len(comps) > 1:
            closed = max(comps, key=lambda m: m.volume if m.is_volume else m.area)
    return closed


# =========================================================================
# SDF grid computation
# =========================================================================
def compute_sdf_grid(mesh, resolution=64, padding=0.1):
    """
    Compute SDF grid from mesh.

    Returns:
        sdf_grid: (R, R, R) float32  — positive=outside, negative=inside
        grid_min: (3,) float32
        grid_max: (3,) float32
    """
    if mesh is None or mesh.is_empty:
        empty = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        return empty, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    bounds = mesh.bounds.astype(np.float32)  # (2, 3)
    grid_min = bounds[0] - padding
    grid_max = bounds[1] + padding

    xs = np.linspace(grid_min[0], grid_max[0], resolution, dtype=np.float32)
    ys = np.linspace(grid_min[1], grid_max[1], resolution, dtype=np.float32)
    zs = np.linspace(grid_min[2], grid_max[2], resolution, dtype=np.float32)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # trimesh: inside -> positive, outside -> negative
    sdf = trimesh.proximity.signed_distance(mesh, points).astype(np.float32)
    # Flip to Scene-Diffuser convention: outside -> positive, inside -> negative
    sdf = -sdf

    sdf_grid = sdf.reshape(resolution, resolution, resolution)
    return sdf_grid, grid_min, grid_max


# =========================================================================
# Export one sample in Scene-Diffuser format
# =========================================================================
def export_sample(
    model, latent, R, t, s,
    out_dir, sample_name, device,
    mesh_resolution=128,
    sdf_resolution=64,
    sdf_padding=0.1,
    sdf_watertight=True,
    sdf_watertight_pitch=0.02,
):
    """
    Save one augmented chair as:
      out_dir/<sample_name>/
        <sample_name>.ply
        <sample_name>.obj
        <sample_name>_sdf.npy
        <sample_name>_sdf_min.npy
        <sample_name>_sdf_max.npy
    """
    sample_dir = Path(out_dir) / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Generate mesh
    mesh = create_mesh(
        model, latent, N=mesh_resolution, max_batch=32**3,
        device=device, R=R, t=t, s=s,
    )
    if mesh is None or mesh.is_empty:
        print(f"  [WARN] Empty mesh: {sample_name}")
        return False

    # Save mesh
    mesh.export(str(sample_dir / f"{sample_name}.ply"))
    mesh.export(str(sample_dir / f"{sample_name}.obj"))

    # Watertight conversion for SDF
    mesh_for_sdf = mesh
    if sdf_watertight:
        try:
            mesh_for_sdf = make_closed_watertight(mesh, pitch=sdf_watertight_pitch)
        except Exception as e:
            print(f"  [WARN] watertight failed for {sample_name}: {e}")
            mesh_for_sdf = mesh

    # Compute & save SDF grid
    sdf_grid, sdf_min, sdf_max = compute_sdf_grid(
        mesh_for_sdf, resolution=sdf_resolution, padding=sdf_padding,
    )
    np.save(str(sample_dir / f"{sample_name}_sdf.npy"), sdf_grid)
    np.save(str(sample_dir / f"{sample_name}_sdf_min.npy"), sdf_min)
    np.save(str(sample_dir / f"{sample_name}_sdf_max.npy"), sdf_max)

    return True


# =========================================================================
# Augmentation operations
# =========================================================================
def remove_armrests(latent, R, t, s):
    lat = latent.clone()
    lat[0, ARM_LEFT] = 0.0
    lat[0, ARM_RIGHT] = 0.0
    return lat, R, t, s


def scale_backrest(latent, R, t, s, thickness_factor=1.0, height_factor=1.0):
    s_new = s.clone()
    s_new[0, BACKREST, 0] *= thickness_factor
    s_new[0, BACKREST, 1] *= height_factor
    return latent, R, t, s_new


def adjust_seat_height(latent, R, t, s, dy=0.0):
    t_new = t.clone()
    t_new[0, SEAT, 1] += dy
    t_new[0, BACKREST, 1] += dy
    t_new[0, ARM_LEFT, 1] += dy
    t_new[0, ARM_RIGHT, 1] += dy
    return latent, R, t_new, s


def scale_legs(latent, R, t, s, height_factor=1.0, thickness_factor=1.0):
    s_new = s.clone()
    for leg in LEGS:
        s_new[0, leg, 1] *= height_factor
        s_new[0, leg, 0] *= thickness_factor
        s_new[0, leg, 2] *= thickness_factor
    return latent, R, t, s_new


def widen_seat(latent, R, t, s, factor=1.0):
    s_new = s.clone()
    s_new[0, SEAT, 2] *= factor
    return latent, R, t, s_new


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def tilt_backrest(latent, R, t, s, angle_deg=0.0, keep_bottom_attached=True):
    if R is None or t is None or s is None:
        return latent, R, t, s

    R_new = R.clone()
    t_new = t.clone()
    theta = np.radians(angle_deg)

    q_delta = torch.tensor(
        [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
        dtype=R.dtype, device=R.device,
    )
    R_new[0, BACKREST] = quaternion_multiply(q_delta, R_new[0, BACKREST])
    R_new[0, BACKREST] = standardize_quaternion(R_new[0, BACKREST].unsqueeze(0)).squeeze(0)

    if keep_bottom_attached:
        half_h = 0.5 * s[0, BACKREST, 1]
        dx = torch.sin(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h
        dy = torch.cos(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h - half_h
        t_new[0, BACKREST, 0] += dx
        t_new[0, BACKREST, 1] += dy

    return latent, R_new, t_new, s


def interpolate_shapes(lat1, R1, t1, s1, lat2, R2, t2, s2, alpha):
    lat = (1 - alpha) * lat1 + alpha * lat2
    R = slerp_quaternion(R1, R2, alpha)
    t = (1 - alpha) * t1 + alpha * t2
    s = (1 - alpha) * s1 + alpha * s2
    return lat, R, t, s


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PartSDF chair augmentation → Scene-Diffuser SDF format"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expdir", default="experiments/chair")
    parser.add_argument("--outdir", default="results/augmented_sdf")
    parser.add_argument("--resolution", type=int, default=128,
                        help="mesh extraction resolution")
    parser.add_argument("--n-source", type=int, default=10,
                        help="number of source chairs to augment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf-resolution", type=int, default=64,
                        help="SDF grid resolution (64 = accurate but slow, 32 = fast)")
    parser.add_argument("--sdf-padding", type=float, default=0.1,
                        help="padding around mesh bounds for SDF grid (meters)")
    parser.add_argument("--sdf-pitch", type=float, default=0.02,
                        help="voxel pitch for watertight conversion")
    parser.add_argument("--no-watertight", action="store_true",
                        help="skip watertight conversion before SDF computation")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    model, latents, poses, specs = load_model(args.expdir, device)
    n_train = latents.num_embeddings

    source_ids = np.linspace(0, n_train - 1, args.n_source, dtype=int).tolist()
    print(f"Source chairs: {source_ids}")

    export_kw = dict(
        device=device,
        mesh_resolution=args.resolution,
        sdf_resolution=args.sdf_resolution,
        sdf_padding=args.sdf_padding,
        sdf_watertight=not args.no_watertight,
        sdf_watertight_pitch=args.sdf_pitch,
    )

    count = 0

    for idx in source_ids:
        lat, R, t, s = get_shape(latents, poses, idx, device)
        prefix = f"chair{idx}"

        # --- Original ---
        name = f"{prefix}_original"
        if export_sample(model, lat, R, t, s, args.outdir, name, **export_kw):
            count += 1

        # --- 1. Remove armrests ---
        lat_m, R_m, t_m, s_m = remove_armrests(lat, R, t, s)
        name = f"{prefix}_no_arms"
        if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
            count += 1

        # --- 2. Thicker backrest ---
        for tf in [1.3, 1.5]:
            lat_m, R_m, t_m, s_m = scale_backrest(lat, R, t, s, thickness_factor=tf)
            name = f"{prefix}_backrest_thick_{tf:.1f}"
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        # --- 3. Taller backrest ---
        for hf in [1.2, 1.4]:
            lat_m, R_m, t_m, s_m = scale_backrest(lat, R, t, s, height_factor=hf)
            name = f"{prefix}_backrest_tall_{hf:.1f}"
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        # --- 3.5. Tilt backrest ---
        for ang in [-15, -10, 10, 15]:
            lat_m, R_m, t_m, s_m = tilt_backrest(lat, R, t, s, angle_deg=ang,
                                                   keep_bottom_attached=True)
            name = f"{prefix}_backrest_tilt_{ang:+d}"
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        # --- 4. Seat height ---
        for dy in [-0.05, 0.05, 0.10]:
            lat_m, R_m, t_m, s_m = adjust_seat_height(lat, R, t, s, dy=dy)
            name = f"{prefix}_seat_dy_{dy:+.2f}".replace("+", "p").replace("-", "m")
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        # --- 5. Leg height ---
        for lh in [0.8, 1.2]:
            lat_m, R_m, t_m, s_m = scale_legs(lat, R, t, s, height_factor=lh)
            name = f"{prefix}_legs_h_{lh:.1f}"
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        # --- 6. Wider seat ---
        for wf in [1.2, 1.4]:
            lat_m, R_m, t_m, s_m = widen_seat(lat, R, t, s, factor=wf)
            name = f"{prefix}_seat_wide_{wf:.1f}"
            if export_sample(model, lat_m, R_m, t_m, s_m, args.outdir, name, **export_kw):
                count += 1

        print(f"  {prefix}: done")

    # --- 7. Interpolations between pairs ---
    pairs = list(itertools.combinations(source_ids[:5], 2))
    for i1, i2 in pairs:
        lat1, R1, t1, s1 = get_shape(latents, poses, i1, device)
        lat2, R2, t2, s2 = get_shape(latents, poses, i2, device)
        for alpha in [0.25, 0.5, 0.75]:
            lat_i, R_i, t_i, s_i = interpolate_shapes(
                lat1, R1, t1, s1, lat2, R2, t2, s2, alpha
            )
            name = f"interp_c{i1}_c{i2}_a{alpha:.2f}"
            if export_sample(model, lat_i, R_i, t_i, s_i, args.outdir, name, **export_kw):
                count += 1

    print(f"\nDone! Generated {count} samples in {args.outdir}/")
    print(f"Each folder contains: .ply, .obj, _sdf.npy, _sdf_min.npy, _sdf_max.npy")


if __name__ == "__main__":
    main()
