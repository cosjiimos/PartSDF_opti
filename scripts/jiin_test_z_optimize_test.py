"""
Test z-latent manipulation on chair 236:

  1. z interpolation with other chairs  → blends detail/style
  2. Per-part safe noise injection       → each part gets noise proportional
     to its training distribution std, so results stay in-distribution

Per-part training distribution std (from 1065 chairs):
  Part 0 (Backrest):  std=0.0202, norm_mean=0.310
  Part 1 (Arm Left):  std=0.0053, norm_mean=0.049
  Part 2 (Arm Right): std=0.0053, norm_mean=0.050
  Part 3 (Leg FR):    std=0.0065, norm_mean=0.096
  Part 4 (Leg BR):    std=0.0064, norm_mean=0.095
  Part 5 (Leg FL):    std=0.0063, norm_mean=0.095
  Part 6 (Leg BL):    std=0.0064, norm_mean=0.095
  Part 7 (Seat):      std=0.0173, norm_mean=0.257

Only saves .obj files for quick visual inspection.

Usage:
  conda activate syncTweedies
  cd /home/user/bodyawarechair/PartSDF
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_test_z_optimize_test.py --device cuda:0
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.mesh import create_mesh
from src.model import get_model, get_part_latents, get_part_poses
from src.utils import set_seed

SCALE_TO_METERS = 0.5

# Per-part training std (from latents_2000.pth, 1065 chairs)
# Noise scaled by this keeps results within the learned distribution
PART_NAMES = ["Backrest", "Arm_L", "Arm_R", "Leg_FR", "Leg_BR", "Leg_FL", "Leg_BL", "Seat"]
PART_STD = [0.0202, 0.0053, 0.0053, 0.0065, 0.0064, 0.0063, 0.0064, 0.0173]


def load_model(expdir, device):
    specs = ws.load_specs(expdir)
    n_parts  = specs["Parts"]["NumParts"]
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

    print(f"Loaded {n_train} chairs, {n_parts} parts, epoch={epoch}")
    return model, latents, poses, n_train


def get_shape(latents, poses, idx, device):
    lat = latents(torch.tensor([idx]).to(device))
    R, t, s = poses(torch.tensor([idx]).to(device))
    return lat, R, t, s


def export_obj(model, lat, R, t, s, name, out_dir, device, mesh_res=128):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mesh = create_mesh(model, lat, N=mesh_res, max_batch=32**3,
                       device=device, R=R, t=t, s=s)
    if mesh is None or mesh.is_empty:
        print(f"  [WARN] empty mesh: {name}")
        return False

    mesh.apply_scale(SCALE_TO_METERS)
    obj_file = out_path / f"{name}.obj"
    mesh.export(obj_file)
    print(f"  saved: {obj_file}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",          default="cuda:0")
    ap.add_argument("--expdir",          default="experiments/chair")
    ap.add_argument("--outdir",          default="results/z_test_236")
    ap.add_argument("--chair-id",        type=int, default=236)
    ap.add_argument("--mesh-resolution", type=int, default=128)
    ap.add_argument("--seed",            type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    model, latents, poses, n_train = load_model(args.expdir, device)

    idx = args.chair_id
    lat0, R0, t0, s0 = get_shape(latents, poses, idx, device)
    print(f"\nSource: chair {idx}")
    print(f"lat0 shape: {lat0.shape}")   # [1, 8, 256]

    export_kw = dict(device=device, mesh_res=args.mesh_resolution, out_dir=args.outdir)

    # ── 0. Original ──────────────────────────────────────────────────────────
    print(f"\n[0] Original chair {idx}")
    export_obj(model, lat0, R0, t0, s0, f"chair{idx}_original", **export_kw)

    # ── 1. Z interpolation with other chairs ─────────────────────────────────
    # Pick a few diverse chairs to blend with
    blend_targets = [0, 100, 500, 800, 1000]
    blend_ratios  = [0.3, 0.5, 0.7]  # how much of target chair's z to mix in

    print(f"\n=== Z Interpolation Tests ===")
    for target_idx in blend_targets:
        if target_idx >= n_train:
            continue
        lat_t, _, _, _ = get_shape(latents, poses, target_idx, device)

        for alpha in blend_ratios:
            # z_new = (1-alpha)*z_source + alpha*z_target
            lat_blend = (1.0 - alpha) * lat0 + alpha * lat_t
            name = f"chair{idx}_blend{target_idx}_a{int(alpha*100):02d}"
            print(f"\n[interp] {name}  (z = {1-alpha:.1f}*chair{idx} + {alpha:.1f}*chair{target_idx})")
            export_obj(model, lat_blend, R0, t0, s0, name, **export_kw)

    # ── 2. Per-part Z interpolation (only backrest from target) ──────────────
    print(f"\n=== Per-part Z Interpolation (backrest only) ===")
    BACKREST = 0
    for target_idx in [0, 500]:
        if target_idx >= n_train:
            continue
        lat_t, _, _, _ = get_shape(latents, poses, target_idx, device)

        lat_part = lat0.clone()
        lat_part[0, BACKREST] = 0.5 * lat0[0, BACKREST] + 0.5 * lat_t[0, BACKREST]
        name = f"chair{idx}_backrest_from{target_idx}"
        print(f"\n[part-interp] {name}  (backrest z = 50/50 mix)")
        export_obj(model, lat_part, R0, t0, s0, name, **export_kw)

    # ── 3. Per-part safe noise (all parts, scaled by training std) ──────────
    # multiplier: 0.5 = subtle, 1.0 = moderate, 2.0 = aggressive (still in-dist)
    print(f"\n=== Per-part Safe Noise (each part scaled by its training std) ===")
    noise_multipliers = [0.5, 1.0, 2.0]

    for mult in noise_multipliers:
        torch.manual_seed(args.seed)
        lat_noisy = lat0.clone()
        for p in range(8):
            safe_scale = PART_STD[p] * mult
            lat_noisy[0, p] = lat0[0, p] + torch.randn_like(lat0[0, p]) * safe_scale
        name = f"chair{idx}_safe_noise_x{str(mult).replace('.','p')}"
        print(f"\n[safe-noise] {name}  (multiplier={mult})")
        for p in range(8):
            print(f"    {PART_NAMES[p]:>10s}: σ = {PART_STD[p] * mult:.4f}")
        export_obj(model, lat_noisy, R0, t0, s0, name, **export_kw)

    # ── 4. Single-part noise (one part at a time) ────────────────────────────
    # Perturb each part individually to see its isolated effect
    print(f"\n=== Single-part Noise (one part at a time, 1.0x std) ===")
    for p in range(8):
        torch.manual_seed(args.seed + p)
        lat_sp = lat0.clone()
        lat_sp[0, p] = lat0[0, p] + torch.randn_like(lat0[0, p]) * PART_STD[p]
        name = f"chair{idx}_part{p}_{PART_NAMES[p]}_noise"
        print(f"\n[single-part] {name}  (σ = {PART_STD[p]:.4f})")
        export_obj(model, lat_sp, R0, t0, s0, name, **export_kw)

    print(f"\nDone! Results in {args.outdir}/")


if __name__ == "__main__":
    main()
