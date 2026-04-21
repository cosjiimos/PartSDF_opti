"""
Generate 9 specific variants of chair 236:

  1. chair236_original                     : original
  2. chair236_legs_short                   : legs height x0.80
  3. chair236_legs_shorter                 : legs height x0.65
  4. chair236_tilt10                        : backrest tilt +10 deg
  5. chair236_tilt10_legs_short            : tilt 10 + legs x0.80
  6. chair236_tilt10_legs_shorter          : tilt 10 + legs x0.65
  7. chair236_tilt20                        : backrest tilt +20 deg
  8. chair236_tilt20_legs_short            : tilt 20 + legs x0.80
  9. chair236_tilt20_legs_shorter          : tilt 20 + legs x0.65

Usage:
  conda activate syncTweedies
  cd /home/user/bodyawarechair/PartSDF
  CUDA_VISIBLE_DEVICES=3 python scripts/jiin_chair236_variants.py --device cuda:0
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
from src.mesh import create_mesh, create_parts
from src.model import get_model, get_part_latents, get_part_poses
from src.primitives import standardize_quaternion
from src.utils import set_seed, get_color_parts

# ── Part indices ──────────────────────────────────────────────────────────────
BACKREST = 0
ARM_LEFT, ARM_RIGHT = 1, 2
LEG_FR, LEG_BR, LEG_FL, LEG_BL = 3, 4, 5, 6
SEAT = 7
LEGS = [LEG_FR, LEG_BR, LEG_FL, LEG_BL]

SCALE_TO_METERS = 0.5   # PartSDF normalized space → real meters


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(expdir, device):
    specs = ws.load_specs(expdir)
    n_parts   = specs["Parts"]["NumParts"]
    part_dim  = specs["Parts"]["LatentDim"]

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
    return model, latents, poses


def get_shape(latents, poses, idx, device):
    lat = latents(torch.tensor([idx]).to(device))
    R, t, s = poses(torch.tensor([idx]).to(device))
    return lat, R, t, s


# ── Augmentation ops ──────────────────────────────────────────────────────────
def shorten_legs_pose(latent, R, t, s, height_factor):
    """
    Shorten legs via pose while forcing overlap with seat for clean blend.

    Strategy:
    1. Shrink s_y by height_factor
    2. Move leg center UP so that leg top reaches seat center (t_y of seat)
       → forces deep overlap so SDF blend keeps the connection solid

    The bottom of the leg rises, the top penetrates into the seat.
    """
    if height_factor >= 1.0 or R is None:
        return latent, R, t, s

    t_new = t.clone()
    s_new = s.clone()

    seat_ty = float(t[0, SEAT, 1])  # seat center

    seat_bottom = float(t[0, SEAT, 1] - s[0, SEAT, 1] / 2.0)

    for p in LEGS:
        old_sy = float(s[0, p, 1])
        new_sy = old_sy * height_factor
        old_ty = float(t[0, p, 1])
        old_top = old_ty + old_sy / 2.0

        # interpolate: more shrinkage → more overlap toward seat_bottom
        # shrink_ratio: 0 when factor=1 (no change), 1 when factor=0 (max)
        shrink_ratio = 1.0 - height_factor
        new_top = old_top + (seat_bottom - old_top) * shrink_ratio
        new_ty = new_top - new_sy / 2.0

        t_new[0, p, 1] = new_ty
        s_new[0, p, 1] = new_sy
        print(f"  leg {p}: t_y {old_ty:.4f}→{new_ty:.4f}  s_y {old_sy:.4f}→{new_sy:.4f}  "
              f"top {old_top:.4f}→{new_top:.4f}  bot={new_ty - new_sy/2:.4f}")

    return latent, R, t_new, s_new


def cut_legs_mesh(mesh, cut_ratio):
    """
    Shorten legs by cutting the bottom of the mesh.
    cut_ratio: fraction of full height to remove from the bottom.
               e.g. 0.20 → removes bottom 20% of bounding box height.
    The open wound is sealed by make_closed_watertight (voxelize+MC) later.
    """
    if mesh is None or mesh.is_empty or cut_ratio <= 0.0:
        return mesh, 0.0

    y_min = float(mesh.bounds[0][1])
    y_max = float(mesh.bounds[1][1])
    height = y_max - y_min
    cut_y = y_min + height * cut_ratio

    print(f"  cut_legs: y∈[{y_min:.3f}, {y_max:.3f}]  cutting below y={cut_y:.3f} ({cut_ratio*100:.0f}%)")

    # keep faces where ALL 3 vertices are above cut_y
    above = mesh.vertices[:, 1] >= cut_y
    face_mask = above[mesh.faces].all(axis=1)
    kept_idx = np.where(face_mask)[0]

    if len(kept_idx) == 0:
        print("  [WARN] no faces kept, skipping cut")
        return mesh, 0.0

    cut = mesh.submesh([kept_idx], append=True)
    cut.remove_unreferenced_vertices()
    removed = y_min - float(cut.bounds[0][1])
    print(f"  cut result: y∈[{cut.bounds[0][1]:.3f}, {cut.bounds[1][1]:.3f}]")
    return cut, removed


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

    # keep bottom of backrest attached to seat
    half_h = 0.5 * s[0, BACKREST, 1]
    dx = torch.sin(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h
    dy = torch.cos(torch.tensor(theta, dtype=t.dtype, device=t.device)) * half_h - half_h
    t_new[0, BACKREST, 0] += dx
    t_new[0, BACKREST, 1] += dy

    return latent, R_new, t_new, s


# ── Export helpers ────────────────────────────────────────────────────────────
def render_mesh_diagonal(mesh, save_path, resolution=(1024, 1024),
                         elev=20, azim=-60):
    """Render mesh from left-front diagonal view and save as PNG.
    PartSDF uses Y-up; matplotlib uses Z-up, so we swap Y↔Z for plotting.
    """
    fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # swap Y-up (PartSDF) → Z-up (matplotlib): (x, y, z) → (x, z, y)
    verts = mesh.vertices.copy()
    verts = verts[:, [0, 2, 1]]  # x, z, y

    faces = mesh.faces

    # subsample if too many faces
    max_faces = 50000
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    tri_verts = verts[faces]

    # face normals for shading
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # two-light shading for clearer appearance
    light1 = np.array([0.5, 0.3, 0.8]); light1 /= np.linalg.norm(light1)
    light2 = np.array([-0.3, -0.5, 0.6]); light2 /= np.linalg.norm(light2)
    shade = 0.6 * np.abs(normals @ light1) + 0.4 * np.abs(normals @ light2)
    shade = 0.15 + 0.85 * shade
    gray = 0.82
    colors = np.stack([shade * gray, shade * gray, shade * gray,
                       np.ones_like(shade)], axis=-1)

    # sort faces by depth for correct z-ordering
    face_centers = tri_verts.mean(axis=1)
    # approximate camera direction from azim/elev
    az_r = np.radians(azim); el_r = np.radians(elev)
    cam_dir = np.array([np.cos(el_r)*np.cos(az_r),
                        np.cos(el_r)*np.sin(az_r),
                        np.sin(el_r)])
    depth = face_centers @ cam_dir
    order = np.argsort(depth)  # back to front
    tri_verts = tri_verts[order]
    colors = colors[order]

    poly = Poly3DCollection(tri_verts, facecolors=colors,
                            edgecolors="none", linewidths=0)
    ax.add_collection3d(poly)

    # set equal-aspect limits
    center = verts.mean(axis=0)
    extent = max(verts.max(axis=0) - verts.min(axis=0)) / 2 * 1.2
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  saved image: {save_path}")


def make_closed_watertight(mesh, pitch=0.02):
    if mesh is None or mesh.is_empty:
        return mesh
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: m.area)
    voxel = mesh.voxelized(pitch)
    closed = voxel.marching_cubes
    closed.apply_transform(voxel.transform)  # convert voxel-index coords → world coords
    closed.remove_unreferenced_vertices()
    try:
        closed.remove_degenerate_faces()
    except Exception:
        pass
    try:
        closed.remove_duplicate_faces()
    except Exception:
        pass
    closed.fix_normals()
    comps = closed.split(only_watertight=False)
    if len(comps) > 1:
        closed = max(comps, key=lambda m: m.volume if m.is_volume else m.area)
    return closed


def compute_sdf_grid(mesh, resolution=64, padding=0.1):
    if mesh is None or mesh.is_empty:
        z = np.zeros((resolution,)*3, dtype=np.float32)
        return z, np.zeros(3, np.float32), np.zeros(3, np.float32)

    bounds  = mesh.bounds.astype(np.float32)
    gmin    = bounds[0] - padding
    gmax    = bounds[1] + padding

    xs = np.linspace(gmin[0], gmax[0], resolution, dtype=np.float32)
    ys = np.linspace(gmin[1], gmax[1], resolution, dtype=np.float32)
    zs = np.linspace(gmin[2], gmax[2], resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sdf = trimesh.proximity.signed_distance(mesh, pts).astype(np.float32)
    sdf = -sdf   # trimesh: inside=+, we want outside=+
    return sdf.reshape(resolution, resolution, resolution), gmin, gmax


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

    # .ply + .obj
    mesh.export(sample_dir / f"{name}.ply")
    mesh.export(sample_dir / f"{name}.obj")
    print(f"  saved mesh: {name}.ply / .obj")

    # render left-diagonal view
    try:
        render_mesh_diagonal(mesh, sample_dir / f"{name}.png")
    except Exception as e:
        print(f"  [WARN] render failed: {e}")

    # PartSDF mesh is already watertight (marching cubes on SDF grid)
    # Compute SDF directly from mesh (world-space)
    print(f"  computing SDF ({sdf_res}^3)…")
    sdf_grid, sdf_min, sdf_max = compute_sdf_grid(mesh, sdf_res, sdf_pad)
    np.save(sample_dir / f"{name}_sdf.npy",     sdf_grid)
    np.save(sample_dir / f"{name}_sdf_min.npy", sdf_min)
    np.save(sample_dir / f"{name}_sdf_max.npy", sdf_max)
    print(f"  saved SDF → {sample_dir}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",         default="cuda:0")
    ap.add_argument("--expdir",         default="experiments/chair")
    ap.add_argument("--outdir",         default="results/chair236_variants_fin")
    ap.add_argument("--chair-id",       type=int, default=236)
    ap.add_argument("--mesh-resolution",type=int, default=128)
    ap.add_argument("--sdf-resolution", type=int, default=64)
    ap.add_argument("--sdf-padding",    type=float, default=0.1)
    ap.add_argument("--wt-pitch",       type=float, default=0.02)
    ap.add_argument("--tilt-small",     type=float, default=5.0,
                    help="'small' tilt angle in degrees (default 5)")
    ap.add_argument("--tilt-large",     type=float, default=15.0,
                    help="'large' tilt angle in degrees (default 15)")
    ap.add_argument("--legs-short",     type=float, default=0.80,
                    help="pose height_factor for short legs: 0.80 = 80%% of original (default 0.80)")
    ap.add_argument("--legs-shorter",   type=float, default=0.60,
                    help="pose height_factor for shorter legs: 0.60 = 60%% of original (default 0.60)")
    ap.add_argument("--seed",           type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    model, latents, poses = load_model(args.expdir, device)

    idx = args.chair_id
    lat0, R0, t0, s0 = get_shape(latents, poses, idx, device)
    print(f"\nSource: chair {idx}")

    export_kw = dict(
        device=device,
        mesh_res=args.mesh_resolution,
        sdf_res=args.sdf_resolution,
        sdf_pad=args.sdf_padding,
        wt_pitch=args.wt_pitch,
        out_dir=args.outdir,
    )

    fac_s  = args.legs_short    # e.g. 0.75
    fac_ss = args.legs_shorter  # e.g. 0.55

    ang_s = args.tilt_small
    ang_l = args.tilt_large

    # variants: (lat, R, t, s, name)
    variants = []

    # 1. original
    variants.append((lat0, R0, t0, s0, f"chair{idx}_original"))

    # 2. legs short (pose: top fixed, bottom rises)
    _, R_s, t_s, s_s = shorten_legs_pose(lat0, R0, t0, s0, fac_s)
    variants.append((lat0, R_s, t_s, s_s, f"chair{idx}_legs_short"))

    # 3. legs shorter
    _, R_ss, t_ss, s_ss = shorten_legs_pose(lat0, R0, t0, s0, fac_ss)
    variants.append((lat0, R_ss, t_ss, s_ss, f"chair{idx}_legs_shorter"))

    # 4. tilt small
    lat_t, R_t, t_t, s_t = tilt_backrest(lat0, R0, t0, s0, ang_s)
    variants.append((lat_t, R_t, t_t, s_t, f"chair{idx}_tilt{int(ang_s)}"))

    # 5. tilt small + legs short
    _, R_ts, t_ts, s_ts = shorten_legs_pose(lat_t, R_t, t_t, s_t, fac_s)
    variants.append((lat_t, R_ts, t_ts, s_ts, f"chair{idx}_tilt{int(ang_s)}_legs_short"))

    # 6. tilt small + legs shorter
    _, R_tss, t_tss, s_tss = shorten_legs_pose(lat_t, R_t, t_t, s_t, fac_ss)
    variants.append((lat_t, R_tss, t_tss, s_tss, f"chair{idx}_tilt{int(ang_s)}_legs_shorter"))

    # 7. tilt large
    lat_T, R_T, t_T, s_T = tilt_backrest(lat0, R0, t0, s0, ang_l)
    variants.append((lat_T, R_T, t_T, s_T, f"chair{idx}_tilt{int(ang_l)}"))

    # 8. tilt large + legs short
    _, R_Ts, t_Ts, s_Ts = shorten_legs_pose(lat_T, R_T, t_T, s_T, fac_s)
    variants.append((lat_T, R_Ts, t_Ts, s_Ts, f"chair{idx}_tilt{int(ang_l)}_legs_short"))

    # 9. tilt large + legs shorter
    _, R_Tss, t_Tss, s_Tss = shorten_legs_pose(lat_T, R_T, t_T, s_T, fac_ss)
    variants.append((lat_T, R_Tss, t_Tss, s_Tss, f"chair{idx}_tilt{int(ang_l)}_legs_shorter"))

    count = 0
    for i, (lat, R, t, s, name) in enumerate(variants, 1):
        print(f"\n[{i}/9] {name}")
        ok = export_variant(model, lat, R, t, s, name=name, **export_kw)
        if ok:
            count += 1

    print(f"\nDone! {count}/9 variants saved to {args.outdir}/")


if __name__ == "__main__":
    main()
