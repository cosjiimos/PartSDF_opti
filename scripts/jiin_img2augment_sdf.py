"""
Image → Trellis 3D → PartSDF reconstruction → Augmentation → Scene-Diffuser SDF export.

Pipeline:
  1. Image → 3D mesh (Trellis, closed mesh)
  2. Quality check + watertight conversion
  3. Normalize to [-0.9, 0.9]³
  4. Generate SDF samples + pseudo part labels
  5. PartSDF reconstruction (latent + pose)
  6. Augmentations (remove arms, scale backrest, tilt, seat height, legs, widen seat)
  7. Export each variant as Scene-Diffuser SDF format

Output per variant:
  <name>/
  ├── <name>.ply
  ├── <name>.obj
  ├── <name>_sdf.npy      # (D,D,D) float32, positive=outside, negative=inside
  ├── <name>_sdf_min.npy   # (3,) float32
  └── <name>_sdf_max.npy   # (3,) float32

Usage:
  conda activate physiopt
  cd /home/user/bodyawarechair/PartSDF
  CUDA_VISIBLE_DEVICES=0 python scripts/jiin_img2augment_sdf.py imgs/0.png --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python scripts/jiin_img2augment_sdf.py imgs/0.png --device cuda:0 --skip-trellis
"""

import os
import sys
import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.loss import get_loss_recon
from src.mesh import create_mesh
from src.model import get_model, get_part_latents, get_part_poses
from src.primitives import standardize_quaternion, slerp_quaternion
from src.reconstruct import reconstruct, reconstruct_parts
from src.utils import set_seed, get_sdf_mesh


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

# PartSDF normalized [-0.9, 0.9]³ → meters (1.8 units ≈ 0.9m chair)
SCALE_TO_METERS = 0.5


# =========================================================================
# Step 1: Image → 3D mesh (Trellis)
# =========================================================================
def image_to_3d(image_path, device="cuda:0"):
    """Generate closed mesh from image via Trellis."""
    from trellis.pipelines import TrellisImageTo3DPipeline

    print(f"[Step 1] Image → 3D mesh via Trellis (device={device})")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    pipeline.to(device)

    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        outputs = pipeline.run(image, seed=0, formats=["mesh"])

    mesh_result = outputs["mesh"]
    if hasattr(mesh_result, "vertices") and isinstance(mesh_result.vertices, torch.Tensor):
        verts = mesh_result.vertices.cpu().numpy()
        faces = mesh_result.faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    elif isinstance(mesh_result, trimesh.Trimesh):
        mesh = mesh_result
    else:
        if hasattr(mesh_result, "geometry"):
            mesh = trimesh.util.concatenate(list(mesh_result.geometry.values()))
        else:
            tmp_path = "/tmp/_trellis_output.glb"
            mesh_result.export(tmp_path)
            loaded = trimesh.load(tmp_path)
            mesh = trimesh.util.concatenate(list(loaded.geometry.values())) \
                if isinstance(loaded, trimesh.Scene) else loaded

    del pipeline, outputs
    torch.cuda.empty_cache()

    print(f"  → {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


# =========================================================================
# Step 1.5: Quality check + watertight
# =========================================================================
def check_and_fix_mesh(mesh, pitch=0.01):
    """Check quality and make watertight if needed."""
    print("[Step 1.5] Mesh quality check")
    print(f"  watertight={mesh.is_watertight}, faces={len(mesh.faces)}")
    extents = mesh.bounding_box.extents
    max_ext = extents.max()
    aspect = extents / max_ext
    print(f"  aspect (X,Y,Z): {aspect[0]:.2f}, {aspect[1]:.2f}, {aspect[2]:.2f}")

    if not mesh.is_watertight:
        print("  → Making watertight via voxelization")
        extent = mesh.bounding_box.extents.max()
        auto_pitch = extent / 100.0
        pitch = min(pitch, auto_pitch)
        voxel = mesh.voxelized(pitch=pitch)
        mesh = voxel.marching_cubes
        print(f"  → After: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}")

    return mesh


# =========================================================================
# Step 2: Normalize
# =========================================================================
def normalize_mesh(mesh, target_range=0.9):
    """Center and scale mesh to [-0.9, 0.9]³."""
    print("[Step 2] Normalizing mesh")
    vertices = mesh.vertices.copy()
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices -= center
    scale = target_range / np.abs(vertices).max()
    vertices *= scale
    mesh_norm = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy())
    print(f"  → Bounds: [{vertices.min():.3f}, {vertices.max():.3f}]")
    return mesh_norm


# =========================================================================
# Step 3: SDF samples + pseudo labels
# =========================================================================
def generate_sdf_samples(mesh, n_samples=250000):
    """Generate DeepSDF-style SDF samples."""
    print("[Step 3] Generating SDF samples")
    surface_pts = mesh.sample(n_samples)
    near_xyz = np.concatenate([
        surface_pts + np.random.normal(scale=sqrt(0.005), size=surface_pts.shape),
        surface_pts + np.random.normal(scale=sqrt(0.0005), size=surface_pts.shape),
    ], axis=0)
    near_sdf = get_sdf_mesh(mesh, near_xyz)
    near_samples = np.concatenate([near_xyz, near_sdf[:, None]], axis=1).astype(np.float32)
    near_samples = near_samples[~np.isnan(near_samples).any(axis=1)]

    n_uniform = n_samples // 10
    unif_xyz = np.random.uniform(-1, 1, (n_uniform, 3))
    unif_sdf = get_sdf_mesh(mesh, unif_xyz)
    unif_samples = np.concatenate([unif_xyz, unif_sdf[:, None]], axis=1).astype(np.float32)
    unif_samples = unif_samples[~np.isnan(unif_samples).any(axis=1)]

    all_samples = np.concatenate([near_samples, unif_samples], axis=0)
    pos = all_samples[all_samples[:, 3] >= 0]
    neg = all_samples[all_samples[:, 3] < 0]
    print(f"  → {len(pos)} positive, {len(neg)} negative samples")
    return {"pos": pos, "neg": neg}


def generate_pseudo_labels(sdf_data, poses, n_parts=8):
    """Assign SDF samples to nearest part via training mean poses."""
    print("[Step 3.5] Generating pseudo part labels")
    pose_w = poses.weight.detach().cpu()
    mean_t = pose_w[:, :, 4:7].mean(0).numpy()
    mean_s = np.clip(pose_w[:, :, 7:10].mean(0).numpy(), 0.01, None)

    def assign(samples):
        xyz = samples[:, :3]
        dists = np.zeros((len(xyz), n_parts))
        for p in range(n_parts):
            diff = xyz - mean_t[p]
            scaled = diff / mean_s[p]
            dists[:, p] = np.sum(scaled ** 2, axis=1)
        return np.argmin(dists, axis=1).astype(np.float32)

    return {"pos": assign(sdf_data["pos"]), "neg": assign(sdf_data["neg"])}


# =========================================================================
# Step 4: PartSDF reconstruction
# =========================================================================
def load_partsdf_model(expdir, device):
    """Load pretrained PartSDF model."""
    print("[Step 4a] Loading PartSDF model")
    specs = ws.load_specs(expdir)
    specs["Device"] = device
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

    try:
        with open(specs["TrainSplit"]) as f:
            n_train = len(json.load(f))
    except (FileNotFoundError, KeyError):
        ckpt = torch.load(
            os.path.join(expdir, "latent", f"latents_{epoch}.pth"), map_location="cpu"
        )
        n_train = ckpt["weight"].shape[0]

    latents = get_part_latents(n_train, n_parts, part_dim,
                               specs.get("LatentBound", None), device=device)
    ws.load_latents(expdir, latents, epoch)

    poses = get_part_poses(n_train, n_parts, freeze=True, device=device, fill_nans=True)
    ws.load_poses(expdir, poses)

    print(f"  → {n_train} training shapes, epoch={epoch}")
    return model, latents, poses, specs


def reconstruct_from_sdf(model, sdf_data, latents, poses, specs, device,
                         n_iters=800, lr=5e-3, label_data=None):
    """2-stage reconstruction: coarse (latent+pose) then refine (latent only)."""
    n_parts = specs["Parts"]["NumParts"]
    part_dim = specs["Parts"]["LatentDim"]
    clampD = specs["ClampingDistance"]
    latent_reg = specs["Parts"].get("LatentRegLambda", 1e-4)
    inter_lambda = specs["Parts"].get("IntersectionLambda", None)
    inter_temp = specs["Parts"].get("IntersectionTemp", 1.0)

    latent_init = latents.weight.detach().clone().mean(0, keepdim=True)
    pose_mean = poses.weight.detach().clone().mean(0, keepdim=True)
    R_init = standardize_quaternion(pose_mean[..., :4])
    t_init = pose_mean[..., 4:7]
    s_init = pose_mean[..., 7:10]

    # Stage 1
    print("[Step 4b] Stage 1: Coarse reconstruction (latent + pose)")
    loss_recon = get_loss_recon(specs["ReconLoss"], reduction="none").to(device)
    out = reconstruct(
        model, sdf_data, n_iters, 8000, lr, loss_recon, latent_reg, clampD,
        latent_init=latent_init, latent_size=part_dim, n_parts=n_parts,
        is_part_sdfnet=True, inter_lambda=inter_lambda, inter_temp=inter_temp,
        rotations=R_init, translations=t_init, scales=s_init, lr_pose=lr,
        max_norm=specs.get("LatentBound", None), verbose=True, device=device,
    )
    err1, latent_s1, R_s1, t_s1, s_s1 = out[0], out[1], out[2], out[3], out[4]
    print(f"  → Stage 1 loss={err1:.6f}")

    if label_data is not None:
        print("[Step 4b] Stage 2: Refine with part supervision (latent only)")
        n_refine = max(n_iters // 2, 200)
        out = reconstruct_parts(
            model, sdf_data, label_data, n_refine, 8000, lr * 0.5,
            loss_fn_recon="l1-hard", recon_lambda=0.5,
            loss_fn_parts="l1-hard", parts_lambda=1.0,
            latent_reg=latent_reg, clampD=clampD,
            latent_init=latent_s1, latent_size=part_dim, n_parts=n_parts,
            is_part_sdfnet=True, inter_lambda=inter_lambda, inter_temp=inter_temp,
            rotations=R_s1, translations=t_s1, scales=s_s1,
            max_norm=specs.get("LatentBound", None), verbose=True, device=device,
        )
        latent = out[1]
        R, t, s = R_s1, t_s1, s_s1  # pose from stage 1
        print(f"  → Stage 2 loss={out[0]:.6f}")
    else:
        latent, R, t, s = latent_s1, R_s1, t_s1, s_s1

    return latent, R, t, s


# =========================================================================
# SDF grid computation
# =========================================================================
def make_closed_watertight(mesh, pitch=0.02):
    if mesh.is_empty:
        return mesh
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: m.area)
    voxel = mesh.voxelized(pitch)
    closed = voxel.marching_cubes
    closed.remove_unreferenced_vertices()
    closed.fix_normals()
    comps = closed.split(only_watertight=False)
    if len(comps) > 1:
        closed = max(comps, key=lambda m: m.volume if m.is_volume else m.area)
    return closed


def compute_sdf_grid(mesh, resolution=64, padding=0.1):
    """SDF grid: positive=outside, negative=inside."""
    if mesh is None or mesh.is_empty:
        empty = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        return empty, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    bounds = mesh.bounds.astype(np.float32)
    grid_min = bounds[0] - padding
    grid_max = bounds[1] + padding

    xs = np.linspace(grid_min[0], grid_max[0], resolution, dtype=np.float32)
    ys = np.linspace(grid_min[1], grid_max[1], resolution, dtype=np.float32)
    zs = np.linspace(grid_min[2], grid_max[2], resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # trimesh: inside→positive, outside→negative. Flip for Scene-Diffuser convention.
    sdf = -trimesh.proximity.signed_distance(mesh, points).astype(np.float32)
    return sdf.reshape(resolution, resolution, resolution), grid_min, grid_max


# =========================================================================
# Export one sample
# =========================================================================
def export_sample(model, latent, R, t, s, out_dir, name, device,
                  mesh_resolution=128, sdf_resolution=64, sdf_padding=0.1,
                  sdf_watertight=True, sdf_pitch=0.02):
    """Export one augmented chair in Scene-Diffuser SDF format (meters)."""
    sample_dir = Path(out_dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    mesh = create_mesh(model, latent, N=mesh_resolution, max_batch=32**3,
                       device=device, R=R, t=t, s=s)
    if mesh is None or mesh.is_empty:
        print(f"  [WARN] Empty mesh: {name}")
        return False

    # Scale to meters
    mesh.apply_scale(SCALE_TO_METERS)

    # Save mesh
    mesh.export(str(sample_dir / f"{name}.ply"))
    mesh.export(str(sample_dir / f"{name}.obj"))

    # Watertight for SDF
    mesh_for_sdf = mesh
    if sdf_watertight:
        try:
            mesh_for_sdf = make_closed_watertight(mesh, pitch=sdf_pitch)
        except Exception as e:
            print(f"  [WARN] watertight failed: {e}")

    sdf_grid, sdf_min, sdf_max = compute_sdf_grid(
        mesh_for_sdf, resolution=sdf_resolution, padding=sdf_padding
    )
    np.save(str(sample_dir / f"{name}_sdf.npy"), sdf_grid)
    np.save(str(sample_dir / f"{name}_sdf_min.npy"), sdf_min)
    np.save(str(sample_dir / f"{name}_sdf_max.npy"), sdf_max)
    return True


# =========================================================================
# Augmentation operations (in normalized space, before mesh generation)
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
    for part in [SEAT, BACKREST, ARM_LEFT, ARM_RIGHT]:
        t_new[0, part, 1] += dy
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


def tilt_backrest(latent, R, t, s, angle_deg=0.0):
    if R is None or t is None or s is None:
        return latent, R, t, s
    R_new, t_new = R.clone(), t.clone()
    theta = np.radians(angle_deg)
    q_delta = torch.tensor(
        [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
        dtype=R.dtype, device=R.device,
    )
    R_new[0, BACKREST] = quaternion_multiply(q_delta, R_new[0, BACKREST])
    R_new[0, BACKREST] = standardize_quaternion(R_new[0, BACKREST].unsqueeze(0)).squeeze(0)
    half_h = 0.5 * s[0, BACKREST, 1]
    t_new[0, BACKREST, 0] += torch.sin(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h
    t_new[0, BACKREST, 1] += torch.cos(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h - half_h
    return latent, R_new, t_new, s


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Image → Trellis → PartSDF → Augmentation → SDF export"
    )
    parser.add_argument("image", help="input image path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expdir", default="experiments/chair")
    parser.add_argument("--outdir", default="results/img2aug_sdf")
    parser.add_argument("--resolution", type=int, default=128,
                        help="mesh extraction resolution")
    parser.add_argument("--recon-iters", type=int, default=800)
    parser.add_argument("--sdf-resolution", type=int, default=64)
    parser.add_argument("--sdf-padding", type=float, default=0.1)
    parser.add_argument("--sdf-pitch", type=float, default=0.02)
    parser.add_argument("--no-watertight", action="store_true")
    parser.add_argument("--skip-trellis", action="store_true",
                        help="skip Trellis, use cached mesh from outdir")
    parser.add_argument("--mesh-path", type=str, default=None,
                        help="use existing mesh instead of Trellis")
    args = parser.parse_args()

    set_seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    # =================================================================
    # Step 1: Image → 3D mesh
    # =================================================================
    trellis_path = os.path.join(args.outdir, "01_trellis_raw.obj")
    if args.mesh_path:
        mesh_raw = trimesh.load(args.mesh_path, force="mesh")
        print(f"[Step 1] Loaded mesh from {args.mesh_path}")
    elif args.skip_trellis and os.path.exists(trellis_path):
        mesh_raw = trimesh.load(trellis_path, force="mesh")
        print(f"[Step 1] Loaded cached mesh from {trellis_path}")
    else:
        mesh_raw = image_to_3d(args.image, device=device)
        mesh_raw.export(trellis_path)

    # =================================================================
    # Step 1.5: Quality check + watertight
    # =================================================================
    mesh_raw = check_and_fix_mesh(mesh_raw)
    mesh_raw.export(os.path.join(args.outdir, "01_watertight.obj"))

    # =================================================================
    # Step 2: Normalize
    # =================================================================
    mesh_norm = normalize_mesh(mesh_raw)
    mesh_norm.export(os.path.join(args.outdir, "02_normalized.obj"))

    # =================================================================
    # Step 3: SDF samples + pseudo labels
    # =================================================================
    sdf_data = generate_sdf_samples(mesh_norm)
    model, latents, poses, specs = load_partsdf_model(args.expdir, device=device)
    label_data = generate_pseudo_labels(sdf_data, poses, n_parts=specs["Parts"]["NumParts"])

    # =================================================================
    # Step 4: PartSDF reconstruction
    # =================================================================
    latent, R, t, s = reconstruct_from_sdf(
        model, sdf_data, latents, poses, specs, device=device,
        n_iters=args.recon_iters, label_data=label_data,
    )

    # Save reconstruction
    recon_mesh = create_mesh(model, latent, N=args.resolution, max_batch=32**3,
                             device=device, R=R, t=t, s=s)
    recon_mesh.export(os.path.join(args.outdir, "03_reconstruction.obj"))
    torch.save({"latent": latent, "R": R, "t": t, "s": s},
               os.path.join(args.outdir, "03_latent_pose.pth"))

    # =================================================================
    # Step 5: Augmentation → SDF export
    # =================================================================
    print("\n[Step 5] Generating augmented variants with SDF export")
    aug_dir = os.path.join(args.outdir, "augmented")

    export_kw = dict(
        device=device,
        mesh_resolution=args.resolution,
        sdf_resolution=args.sdf_resolution,
        sdf_padding=args.sdf_padding,
        sdf_watertight=not args.no_watertight,
        sdf_pitch=args.sdf_pitch,
    )

    prefix = "chair_img"
    count = 0

    # Original reconstruction
    if export_sample(model, latent, R, t, s, aug_dir, f"{prefix}_original", **export_kw):
        count += 1

    # 1. Remove armrests
    lat_m, R_m, t_m, s_m = remove_armrests(latent, R, t, s)
    if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir, f"{prefix}_no_arms", **export_kw):
        count += 1

    # 2. Thicker backrest
    for tf in [1.3, 1.5]:
        lat_m, R_m, t_m, s_m = scale_backrest(latent, R, t, s, thickness_factor=tf)
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir,
                         f"{prefix}_backrest_thick_{tf:.1f}", **export_kw):
            count += 1

    # 3. Taller backrest
    for hf in [1.2, 1.4]:
        lat_m, R_m, t_m, s_m = scale_backrest(latent, R, t, s, height_factor=hf)
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir,
                         f"{prefix}_backrest_tall_{hf:.1f}", **export_kw):
            count += 1

    # 3.5. Tilt backrest
    for ang in [-15, -10, 10, 15]:
        lat_m, R_m, t_m, s_m = tilt_backrest(latent, R, t, s, angle_deg=ang)
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir,
                         f"{prefix}_backrest_tilt_{ang:+d}", **export_kw):
            count += 1

    # 4. Seat height
    for dy in [-0.05, 0.05, 0.10]:
        lat_m, R_m, t_m, s_m = adjust_seat_height(latent, R, t, s, dy=dy)
        name = f"{prefix}_seat_dy_{dy:+.2f}".replace("+", "p").replace("-", "m")
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir, name, **export_kw):
            count += 1

    # 5. Leg height
    for lh in [0.8, 1.2]:
        lat_m, R_m, t_m, s_m = scale_legs(latent, R, t, s, height_factor=lh)
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir,
                         f"{prefix}_legs_h_{lh:.1f}", **export_kw):
            count += 1

    # 6. Wider seat
    for wf in [1.2, 1.4]:
        lat_m, R_m, t_m, s_m = widen_seat(latent, R, t, s, factor=wf)
        if export_sample(model, lat_m, R_m, t_m, s_m, aug_dir,
                         f"{prefix}_seat_wide_{wf:.1f}", **export_kw):
            count += 1

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Pipeline complete! {count} augmented variants exported.")
    print(f"  Intermediate: {args.outdir}/01_*.obj, 02_*.obj, 03_*.obj")
    print(f"  Augmented:    {aug_dir}/")
    print(f"  Each folder:  .ply, .obj, _sdf.npy, _sdf_min.npy, _sdf_max.npy")
    print(f"  Scale: meters (chair height ≈ 0.9m)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
