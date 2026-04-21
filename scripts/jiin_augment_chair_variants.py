"""
Generate augmented variants for one or more chairs from the training set.

Variants per chair (up to 9, or 12 if armrests detected):
  1.  chair{N}_original
  2.  chair{N}_legs_short
  3.  chair{N}_legs_shorter
  4.  chair{N}_tilt{S}
  5.  chair{N}_tilt{S}_legs_short
  6.  chair{N}_tilt{S}_legs_shorter
  7.  chair{N}_tilt{L}
  8.  chair{N}_tilt{L}_legs_short
  9.  chair{N}_tilt{L}_legs_shorter
  + if armrests detected:
  10. chair{N}_no_arms
  11. chair{N}_no_arms_legs_short
  12. chair{N}_no_arms_legs_shorter

Usage examples:
  # Single chair
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chair_variants.py --chair-id 236

  # Multiple specific IDs
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chair_variants.py --chair-ids 100 236 500

  # Range
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chair_variants.py --chair-range 0 50

  # Range with step
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_augment_chair_variants.py --chair-range 0 1000 --chair-step 50
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.mesh import create_mesh
from src.model import get_model, get_part_latents, get_part_poses
from src.primitives import standardize_quaternion
from src.utils import set_seed

# ── Part indices ──────────────────────────────────────────────────────────────
BACKREST = 0
ARM_LEFT, ARM_RIGHT = 1, 2
LEG_FR, LEG_BR, LEG_FL, LEG_BL = 3, 4, 5, 6
SEAT = 7
LEGS = [LEG_FR, LEG_BR, LEG_FL, LEG_BL]
ARMS = [ARM_LEFT, ARM_RIGHT]

SCALE_TO_METERS = 0.5   # PartSDF normalized space → real meters

# threshold: arm scale X > this → likely a real armrest
ARM_SCALE_THRESHOLD = 0.20


# ── Model loading ─────────────────────────────────────────────────────────────
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


# ── Armrest detection ─────────────────────────────────────────────────────────
def has_armrests(s, threshold=ARM_SCALE_THRESHOLD):
    """Return True if the chair's arm parts have significant scale (real armrests)."""
    s_left  = float(s[0, ARM_LEFT,  0])   # X-scale of left arm
    s_right = float(s[0, ARM_RIGHT, 0])   # X-scale of right arm
    return (s_left > threshold) or (s_right > threshold)


# ── Augmentation ops ──────────────────────────────────────────────────────────
def remove_armrests(latent, R, t, s):
    """Zero-out arm part scales → arms become invisible in SDF blend."""
    s_new = s.clone()
    for p in ARMS:
        s_new[0, p, :] = 1e-4   # near-zero → SDF contribution negligible
    return latent, R, t, s_new


def shorten_legs_pose(latent, R, t, s, height_factor):
    """Shorten legs via pose: top fixed at original position, bottom rises."""
    if height_factor >= 1.0 or R is None:
        return latent, R, t, s

    t_new = t.clone()
    s_new = s.clone()

    seat_bottom = float(t[0, SEAT, 1] - s[0, SEAT, 1] / 2.0)

    for p in LEGS:
        old_sy  = float(s[0, p, 1])
        new_sy  = old_sy * height_factor
        old_ty  = float(t[0, p, 1])
        old_top = old_ty + old_sy / 2.0

        shrink_ratio = 1.0 - height_factor
        new_top = old_top + (seat_bottom - old_top) * shrink_ratio
        new_ty  = new_top - new_sy / 2.0

        t_new[0, p, 1] = new_ty
        s_new[0, p, 1] = new_sy

    return latent, R, t_new, s_new


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

    R_new = R.clone()
    t_new = t.clone()
    theta = np.radians(angle_deg)

    q_delta = torch.tensor(
        [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
        dtype=R.dtype, device=R.device,
    )
    R_new[0, BACKREST] = quaternion_multiply(q_delta, R_new[0, BACKREST])
    R_new[0, BACKREST] = standardize_quaternion(
        R_new[0, BACKREST].unsqueeze(0)
    ).squeeze(0)

    half_h = 0.5 * s[0, BACKREST, 1]
    dx = torch.sin(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h
    dy = torch.cos(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h - half_h
    t_new[0, BACKREST, 0] += dx
    t_new[0, BACKREST, 1] += dy

    return latent, R_new, t_new, s


# ── Export helpers ────────────────────────────────────────────────────────────
def render_mesh_diagonal(mesh, save_path, resolution=(1024, 1024), elev=20, azim=-60):
    """Render mesh from left-front diagonal view and save as PNG."""
    fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # swap Y-up (PartSDF) → Z-up (matplotlib)
    verts = mesh.vertices[:, [0, 2, 1]].copy()
    faces = mesh.faces

    max_faces = 50000
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    tri_verts = verts[faces]
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms

    light1 = np.array([0.5, 0.3, 0.8]); light1 /= np.linalg.norm(light1)
    light2 = np.array([-0.3, -0.5, 0.6]); light2 /= np.linalg.norm(light2)
    shade  = 0.6 * np.abs(normals @ light1) + 0.4 * np.abs(normals @ light2)
    shade  = 0.15 + 0.85 * shade
    gray   = 0.82
    colors = np.stack([shade*gray, shade*gray, shade*gray, np.ones_like(shade)], axis=-1)

    az_r = np.radians(azim); el_r = np.radians(elev)
    cam_dir = np.array([np.cos(el_r)*np.cos(az_r), np.cos(el_r)*np.sin(az_r), np.sin(el_r)])
    depth = tri_verts.mean(axis=1) @ cam_dir
    order = np.argsort(depth)
    tri_verts = tri_verts[order]; colors = colors[order]

    poly = Poly3DCollection(tri_verts, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(poly)

    center = verts.mean(axis=0)
    extent = max(verts.max(axis=0) - verts.min(axis=0)) / 2 * 1.2
    ax.set_xlim(center[0]-extent, center[0]+extent)
    ax.set_ylim(center[1]-extent, center[1]+extent)
    ax.set_zlim(center[2]-extent, center[2]+extent)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  saved image: {save_path}")


def compute_sdf_grid_gpu(model, lat, R, t, s, mesh, device, resolution=64, padding=0.1,
                         max_batch=32**3):
    """Compute SDF grid directly from PartSDF model on GPU.

    Grid bounds are derived from the actual mesh bounds (meters) so the grid
    tightly wraps the chair with only `padding` margin. This maximizes
    resolution where it matters (near the surface).

    Convention: positive=outside, negative=inside (PartSDF native).
    """
    # Mesh is already in meters (SCALE_TO_METERS applied).
    # Convert mesh bounds back to normalized space for model query.
    mesh_min = mesh.bounds[0].astype(np.float32)  # meters
    mesh_max = mesh.bounds[1].astype(np.float32)

    # World-space grid bounds (meters): tight around mesh + padding
    gmin = mesh_min - padding
    gmax = mesh_max + padding

    # Convert to normalized space for model query: meters / SCALE_TO_METERS
    norm_min = gmin / SCALE_TO_METERS
    norm_max = gmax / SCALE_TO_METERS

    # Clamp to [-1, 1] (model's valid range)
    norm_min = np.clip(norm_min, -1.0, 1.0)
    norm_max = np.clip(norm_max, -1.0, 1.0)

    xs = torch.linspace(float(norm_min[0]), float(norm_max[0]), resolution)
    ys = torch.linspace(float(norm_min[1]), float(norm_max[1]), resolution)
    zs = torch.linspace(float(norm_min[2]), float(norm_max[2]), resolution)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)

    # Query model in batches: model(part_lat, xyz, R=, t=, s=)
    sdf_vals = []
    with torch.no_grad():
        for i in range(0, len(grid_pts), max_batch):
            batch = grid_pts[i:i+max_batch].unsqueeze(0)  # [1, B, 3]
            pred = model(lat, batch, R=R, t=t, s=s)       # [1, B, 1]
            pred = pred.reshape(-1)
            sdf_vals.append(pred.cpu())

    sdf = torch.cat(sdf_vals, dim=0).numpy().astype(np.float32)

    # SDF values are in normalized space — scale to meters to match mesh
    sdf = sdf * SCALE_TO_METERS
    sdf_grid = sdf.reshape(resolution, resolution, resolution)

    return sdf_grid, gmin.astype(np.float32), gmax.astype(np.float32)


def export_variant(model, lat, R, t, s, name, out_dir, device,
                   mesh_res=128, sdf_res=64, sdf_pad=0.1, wt_pitch=0.02):
    sample_dir = Path(out_dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    mesh = create_mesh(model, lat, N=mesh_res, max_batch=32**3,
                       device=device, R=R, t=t, s=s)
    if mesh is None or mesh.is_empty:
        print(f"  [WARN] empty mesh: {name}")
        return False

    mesh.apply_scale(SCALE_TO_METERS)

    mesh.export(sample_dir / f"{name}.ply")
    mesh.export(sample_dir / f"{name}.obj")
    print(f"  saved mesh: {name}.ply / .obj")

    try:
        render_mesh_diagonal(mesh, sample_dir / f"{name}.png")
    except Exception as e:
        print(f"  [WARN] render failed: {e}")

    print(f"  computing SDF ({sdf_res}^3) on GPU…")
    sdf_grid, sdf_min, sdf_max = compute_sdf_grid_gpu(model, lat, R, t, s, mesh, device,
                                                        resolution=sdf_res, padding=sdf_pad)
    np.save(sample_dir / f"{name}_sdf.npy",     sdf_grid)
    np.save(sample_dir / f"{name}_sdf_min.npy", sdf_min)
    np.save(sample_dir / f"{name}_sdf_max.npy", sdf_max)
    print(f"  saved SDF → {sample_dir}")
    return True


def build_variants(idx, lat0, R0, t0, s0, fac_s, fac_ss, ang_s, ang_l):
    """Build list of (lat, R, t, s, name) for one chair."""
    variants = []

    # 1-3: original + leg variants
    variants.append((lat0, R0, t0, s0, f"chair{idx}_original"))
    _, R_s, t_s, s_s   = shorten_legs_pose(lat0, R0, t0, s0, fac_s)
    variants.append((lat0, R_s,  t_s,  s_s,  f"chair{idx}_legs_short"))
    _, R_ss, t_ss, s_ss = shorten_legs_pose(lat0, R0, t0, s0, fac_ss)
    variants.append((lat0, R_ss, t_ss, s_ss, f"chair{idx}_legs_shorter"))

    # 4-6: tilt small
    lat_t, R_t, t_t, s_t = tilt_backrest(lat0, R0, t0, s0, ang_s)
    variants.append((lat_t, R_t, t_t, s_t, f"chair{idx}_tilt{int(ang_s)}"))
    _, R_ts,  t_ts,  s_ts  = shorten_legs_pose(lat_t, R_t, t_t, s_t, fac_s)
    variants.append((lat_t, R_ts,  t_ts,  s_ts,  f"chair{idx}_tilt{int(ang_s)}_legs_short"))
    _, R_tss, t_tss, s_tss = shorten_legs_pose(lat_t, R_t, t_t, s_t, fac_ss)
    variants.append((lat_t, R_tss, t_tss, s_tss, f"chair{idx}_tilt{int(ang_s)}_legs_shorter"))

    # 7-9: tilt large
    lat_T, R_T, t_T, s_T = tilt_backrest(lat0, R0, t0, s0, ang_l)
    variants.append((lat_T, R_T, t_T, s_T, f"chair{idx}_tilt{int(ang_l)}"))
    _, R_Ts,  t_Ts,  s_Ts  = shorten_legs_pose(lat_T, R_T, t_T, s_T, fac_s)
    variants.append((lat_T, R_Ts,  t_Ts,  s_Ts,  f"chair{idx}_tilt{int(ang_l)}_legs_short"))
    _, R_Tss, t_Tss, s_Tss = shorten_legs_pose(lat_T, R_T, t_T, s_T, fac_ss)
    variants.append((lat_T, R_Tss, t_Tss, s_Tss, f"chair{idx}_tilt{int(ang_l)}_legs_shorter"))

    # 10-12: no_arms (only if armrests detected)
    # if has_armrests(s0):
    #     lat_a, R_a, t_a, s_a = remove_armrests(lat0, R0, t0, s0)
    #     variants.append((lat_a, R_a, t_a, s_a, f"chair{idx}_no_arms"))
    #     _, R_as,  t_as,  s_as  = shorten_legs_pose(lat_a, R_a, t_a, s_a, fac_s)
    #     variants.append((lat_a, R_as,  t_as,  s_as,  f"chair{idx}_no_arms_legs_short"))
    #     _, R_ass, t_ass, s_ass = shorten_legs_pose(lat_a, R_a, t_a, s_a, fac_ss)
    #     variants.append((lat_a, R_ass, t_ass, s_ass, f"chair{idx}_no_arms_legs_shorter"))

    return variants


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",          default="cuda:0")
    ap.add_argument("--expdir",          default="experiments/chair")
    ap.add_argument("--outdir",          default="results/augmented_variants")
    ap.add_argument("--mesh-resolution", type=int,   default=128)
    ap.add_argument("--sdf-resolution",  type=int,   default=64)
    ap.add_argument("--sdf-padding",     type=float, default=0.1)
    ap.add_argument("--wt-pitch",        type=float, default=0.02)
    ap.add_argument("--tilt-small",      type=float, default=5.0)
    ap.add_argument("--tilt-large",      type=float, default=15.0)
    ap.add_argument("--legs-short",      type=float, default=0.80)
    ap.add_argument("--legs-shorter",    type=float, default=0.60)
    ap.add_argument("--seed",            type=int,   default=0)

    # chair selection (mutually exclusive)
    sel = ap.add_mutually_exclusive_group(required=False)
    sel.add_argument("--chair-id",    type=int,
                     help="single chair index (default: 236)")
    sel.add_argument("--chair-ids",   type=int, nargs="+",
                     help="multiple chair indices, e.g. --chair-ids 100 236 500")
    sel.add_argument("--chair-range", type=int, nargs=2, metavar=("START", "END"),
                     help="range [START, END) e.g. --chair-range 0 100")

    ap.add_argument("--chair-step",   type=int, default=1,
                    help="step for --chair-range (default 1)")

    args = ap.parse_args()
    set_seed(args.seed)

    model, latents, poses, n_train = load_model(args.expdir, args.device)

    # resolve chair list
    if args.chair_ids is not None:
        chair_list = args.chair_ids
    elif args.chair_range is not None:
        start, end = args.chair_range
        end = min(end, n_train)
        chair_list = list(range(start, end, args.chair_step))
    elif args.chair_id is not None:
        chair_list = [args.chair_id]
    else:
        chair_list = [236]

    # clamp to valid range
    chair_list = [i for i in chair_list if 0 <= i < n_train]
    print(f"\nProcessing {len(chair_list)} chair(s): {chair_list}")

    export_kw = dict(
        device=args.device,
        mesh_res=args.mesh_resolution,
        sdf_res=args.sdf_resolution,
        sdf_pad=args.sdf_padding,
        wt_pitch=args.wt_pitch,
        out_dir=args.outdir,
    )

    total_ok = 0
    for chair_idx in chair_list:
        lat0, R0, t0, s0 = get_shape(latents, poses, chair_idx, args.device)
        arm_info = "has armrests" if has_armrests(s0) else "no armrests"
        print(f"\n{'='*60}")
        print(f"Chair {chair_idx}  ({arm_info})")
        print(f"{'='*60}")

        variants = build_variants(
            chair_idx, lat0, R0, t0, s0,
            fac_s=args.legs_short,
            fac_ss=args.legs_shorter,
            ang_s=args.tilt_small,
            ang_l=args.tilt_large,
        )

        n_ok = 0
        for i, (lat, R, t, s, name) in enumerate(variants, 1):
            print(f"\n[{i}/{len(variants)}] {name}")
            ok = export_variant(model, lat, R, t, s, name=name, **export_kw)
            if ok:
                n_ok += 1

        print(f"\nChair {chair_idx}: {n_ok}/{len(variants)} variants saved")
        total_ok += n_ok

    print(f"\n{'='*60}")
    print(f"Done! Total {total_ok} variants saved to {args.outdir}/")


if __name__ == "__main__":
    main()
