#!/usr/bin/env python3
"""Label Vega markers by chopstick, estimate both ends of each chopstick, and render them in MuJoCo.

Input:  a session folder from polaris_vega/vega_realsense_side_by_side.py (markers.csv,
        metadata.json, optionally rgb.mp4 + video_timestamps.csv).
Output: marker_labels.csv   - which chopstick / marker slot every Vega marker belongs to
        chopstick_ends.csv  - 3D tip and back-end positions of both chopsticks (Vega frame, mm)
        chopsticks.mp4      - MuJoCo rendering, one color per chopstick, ends labeled
        chopsticks_side_by_side.mp4 - RealSense RGB | MuJoCo (if rgb.mp4 exists)

Marker positions are measured along the shaft from the tip (thin, eating end), in mm.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import cv2
import mujoco
import numpy as np

# Same display transform as vega_realsense_side_by_side.py: Vega Y->X, -Z->Y, -X->Z.
VEGA_TO_MUJOCO = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)

# name -> marker distances from the tip (mm), total length (mm), RGBA
CHOPSTICKS = {
    'chopstick_A': {'markers': (52.0, 187.0, 227.0), 'length': 241.3, 'rgba': (0.95, 0.25, 0.2, 1)},
    'chopstick_B': {'markers': (76.0, 262.0), 'length': 241.3, 'rgba': (0.2, 0.45, 1.0, 1)},
}
MISSING_PENALTY_MM = 25.0   # cost of leaving a marker slot unmatched
CONTINUITY_BONUS_MM = 3.0   # reward for keeping a Vega marker_id in the same slot as last frame


def load_frames(path):
    frames = collections.OrderedDict()
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            frame = frames.setdefault(int(row['vega_frame']),
                                      {'t': float(row['host_elapsed_s']), 'points': {}})
            if row['marker_id']:
                frame['points'][int(row['marker_id'])] = np.array(
                    [float(row['x_mm']), float(row['y_mm']), float(row['z_mm'])])
    return frames


def chopstick_candidates(points, offsets, tol, previous, name):
    """All ways to map a chopstick's marker slots to detected ids (None = not seen), with cost."""
    ids = list(points)
    options = []
    for combo in itertools.product([None] + ids, repeat=len(offsets)):
        used = [i for i in combo if i is not None]
        if len(used) < 2 or len(set(used)) != len(used):
            continue
        cost = MISSING_PENALTY_MM * (len(offsets) - len(used))
        ok = True
        for (a, ia), (b, ib) in itertools.combinations(zip(offsets, combo), 2):
            if ia is None or ib is None:
                continue
            err = abs(np.linalg.norm(points[ia] - points[ib]) - abs(a - b))
            if err > tol:
                ok = False
                break
            cost += err
        if ok:
            cost -= CONTINUITY_BONUS_MM * sum(previous.get(i) == (name, k) for k, i in enumerate(combo))
            options.append((cost, combo))
    return options


def label_frame(points, tol, previous):
    """Best joint assignment of detected marker ids to both chopsticks: {id: (chopstick, slot)}."""
    names = list(CHOPSTICKS)
    per_stick = [chopstick_candidates(points, CHOPSTICKS[n]['markers'], tol, previous, n)
                 + [(MISSING_PENALTY_MM * len(CHOPSTICKS[n]['markers']), (None,) * len(CHOPSTICKS[n]['markers']))]
                 for n in names]
    best = None
    for choice in itertools.product(*per_stick):
        used = [i for _, combo in choice for i in combo if i is not None]
        if len(set(used)) != len(used):
            continue
        cost = sum(c for c, _ in choice)
        if best is None or cost < best[0]:
            best = (cost, choice)
    return {i: (name, k) for name, (_, combo) in zip(names, best[1])
            for k, i in enumerate(combo) if i is not None}


def fit_ends(positions, offsets, length):
    """Fit a rigid line p(s) = origin + s * axis to markers at known shaft offsets; return tip, back."""
    P, s = np.array(positions), np.array(offsets)
    s_c, P_c = s - s.mean(), P - P.mean(axis=0)
    axis = (s_c[:, None] * P_c).sum(axis=0)
    axis /= np.linalg.norm(axis)
    origin = (P - s[:, None] * axis).mean(axis=0)
    return origin, origin + length * axis


def realsense_camera_xml(calib):
    """MuJoCo camera at the calibrated RealSense pose (see calibrate_realsense_to_vega.py)."""
    R = np.array(calib['R'])
    t = np.array(calib['t_mm'])
    pos = VEGA_TO_MUJOCO @ (-R.T @ t) / 1000
    x_axis = VEGA_TO_MUJOCO @ R[0]
    y_axis = VEGA_TO_MUJOCO @ -R[1]
    fovy = math.degrees(2 * math.atan(calib['height'] / (2 * calib['fy'])))
    fmt = lambda v: ' '.join(f'{c:.6f}' for c in v)
    return (f'<camera name="realsense" pos="{fmt(pos)}" '
            f'xyaxes="{fmt(x_axis)} {fmt(y_axis)}" fovy="{fovy:.4f}"/>')


def to_mj(p_mm):
    return VEGA_TO_MUJOCO @ p_mm / 1000


def add_geom(scene, gtype, size, pos, rgba):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.array(size, float), np.array(pos, float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    scene.ngeom += 1
    return g


def project(scene, point, width, height):
    """Pixel coordinates of a MuJoCo world point in the rendered image (None if behind camera)."""
    gl = scene.camera[0]
    forward, up = np.array(gl.forward), np.array(gl.up)
    v = point - np.array(gl.pos)
    depth = v @ forward
    if depth <= 0:
        return None
    x = (v @ np.cross(forward, up)) * gl.frustum_near / depth
    y = (v @ up) * gl.frustum_near / depth
    # frustum_width is left at 0 by mjv_updateScene; MuJoCo derives it from the aspect ratio.
    half_width = (gl.frustum_top - gl.frustum_bottom) / 2 * width / height
    u = (x - gl.frustum_center + half_width) / (2 * half_width) * width
    w = (gl.frustum_top - y) / (gl.frustum_top - gl.frustum_bottom) * height
    return np.array([u, w])


def draw_chopsticks(scene, ends, markers):
    for name, (tip, back) in ends.items():
        rgba = CHOPSTICKS[name]['rgba']
        g = add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, (0, 0, 0), (0, 0, 0), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, to_mj(tip), to_mj(back))
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), to_mj(tip), rgba)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), to_mj(back), rgba)
    for name, p in markers:
        rgba = CHOPSTICKS[name]['rgba'] if name else (0.6, 0.6, 0.6, 1)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.009, 0, 0), to_mj(p), rgba[:3] + (0.45,))


def put_text(img, text, org, color, scale=0.45):
    org = tuple(int(c) for c in org)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def label_ends(img, scene, ends):
    """Name each end next to it in the image, nudging labels apart when they would overlap."""
    placed = []
    for name, pair in ends.items():
        bgr = tuple(int(255 * c) for c in CHOPSTICKS[name]['rgba'][2::-1])
        for end, p in zip(('tip', 'back'), pair):
            uv = project(scene, to_mj(p), img.shape[1], img.shape[0])
            if uv is None:
                continue
            pos = uv + (12, 5)
            while any(abs(pos[1] - q[1]) < 16 and abs(pos[0] - q[0]) < 80 for q in placed):
                pos[1] += 16
            placed.append(pos.copy())
            put_text(img, f"{name.split('_')[1]} {end}", pos, bgr, 0.5)


def overlay_text(img, frame_index, ends):
    lines = [f'Vega frame {frame_index}']
    for name in CHOPSTICKS:
        bgr = tuple(int(255 * c) for c in CHOPSTICKS[name]['rgba'][2::-1])
        if name not in ends:
            lines.append((f'{name}: not tracked', bgr))
            continue
        for end, p in zip(('tip', 'back'), ends[name]):
            lines.append((f'{name} {end}: ({p[0]:7.1f}, {p[1]:7.1f}, {p[2]:8.1f}) mm', bgr))
    y = 22
    for line in lines:
        text, color = line if isinstance(line, tuple) else (line, (255, 255, 255))
        put_text(img, text, (10, y), color)
        y += 18


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session', help='Recording folder containing markers.csv')
    p.add_argument('--out', help='Output folder (default: the session folder)')
    p.add_argument('--tol-mm', type=float, default=8.0,
                   help='Max error between measured and expected inter-marker distance')
    p.add_argument('--length-a-mm', type=float, default=CHOPSTICKS['chopstick_A']['length'])
    p.add_argument('--length-b-mm', type=float, default=CHOPSTICKS['chopstick_B']['length'],
                   help='Tip-to-back length of the 2-marker chopstick (not in the drawing; '
                        'defaults to the 3-marker chopstick length)')
    p.add_argument('--realsense-view', action='store_true',
                   help='Render from the calibrated RealSense pose instead of a close-up camera that '
                        'follows the chopsticks (looking from the RealSense direction when calibrated)')
    p.add_argument('--distance', type=float, default=0.6, help='Follow-camera distance (m)')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=float, help='Video fps (default: vega_hz from metadata)')
    args = p.parse_args()
    CHOPSTICKS['chopstick_A']['length'] = args.length_a_mm
    CHOPSTICKS['chopstick_B']['length'] = args.length_b_mm

    session = Path(args.session).expanduser()
    out = Path(args.out).expanduser() if args.out else session
    out.mkdir(parents=True, exist_ok=True)
    meta_path = session / 'metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fps = args.fps or meta.get('vega_hz_requested', 30)
    frames = load_frames(session / 'markers.csv')

    # 1) Label markers and estimate ends, frame by frame.
    results, previous = [], {}
    id_votes = collections.defaultdict(collections.Counter)
    for index, frame in frames.items():
        labels = label_frame(frame['points'], args.tol_mm, previous)
        previous = labels
        ends = {}
        for name, spec in CHOPSTICKS.items():
            seen = sorted((k, i) for i, (n, k) in labels.items() if n == name)
            if len(seen) >= 2:
                ends[name] = fit_ends([frame['points'][i] for _, i in seen],
                                      [spec['markers'][k] for k, _ in seen], spec['length'])
        for i, lab in labels.items():
            id_votes[i][lab] += 1
        results.append((index, frame, labels, ends))

    with (out / 'marker_labels.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vega_frame', 'host_elapsed_s', 'marker_id', 'chopstick',
                    'marker_from_tip_mm', 'x_mm', 'y_mm', 'z_mm'])
        for index, frame, labels, _ in results:
            for i, pt in sorted(frame['points'].items()):
                name, k = labels.get(i, ('unassigned', None))
                w.writerow([index, f"{frame['t']:.6f}", i, name,
                            '' if k is None else CHOPSTICKS[name]['markers'][k], *np.round(pt, 2)])

    with (out / 'chopstick_ends.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vega_frame', 'host_elapsed_s', 'chopstick', 'end', 'x_mm', 'y_mm', 'z_mm', 'markers_used'])
        for index, frame, labels, ends in results:
            for name in CHOPSTICKS:
                n_used = sum(n == name for n, _ in labels.values())
                for e, end in enumerate(('tip', 'back')):
                    xyz = np.round(ends[name][e], 2) if name in ends else ('', '', '')
                    w.writerow([index, f"{frame['t']:.6f}", name, end, *xyz, n_used])

    # 2) Render.
    calib = meta.get('camera_calibration')
    width, height = (calib['width'], calib['height']) if calib else (args.width, args.height)
    if args.realsense_view and not calib:
        p.error('--realsense-view needs camera_calibration in metadata.json')
    model = mujoco.MjModel.from_xml_string(f'''<mujoco model="chopsticks">
      <visual><global offwidth="{width}" offheight="{height}"/><map znear="0.01"/></visual>
      <worldbody>
        <light pos="0 1 3" dir="0 -.3 -1" directional="true"/>
        {realsense_camera_xml(calib) if calib else ''}
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if args.realsense_view:
        cam = 'realsense'
    else:
        all_ends = [to_mj(q) for *_, e in results for pair in e.values() for q in pair]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = np.median(all_ends, axis=0) if all_ends else 0
        cam.distance = args.distance
        cam.azimuth, cam.elevation = 135, -20
        if calib:
            R, t = np.array(calib['R']), np.array(calib['t_mm'])
            forward = cam.lookat - VEGA_TO_MUJOCO @ (-R.T @ t) / 1000
            forward /= np.linalg.norm(forward)
            cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
            cam.elevation = math.degrees(math.asin(forward[2]))

    rgb_frames = {}
    if (session / 'rgb.mp4').exists() and (session / 'video_timestamps.csv').exists():
        with (session / 'video_timestamps.csv').open(newline='') as f:
            rgb_frames = {int(r['video_frame']): int(r['rgb_frame_used'])
                          for r in csv.DictReader(f) if r['rgb_frame_used'] not in ('', '-1')}
    rgb_cap = cv2.VideoCapture(str(session / 'rgb.mp4')) if rgb_frames else None
    rgb_pos, rgb_img = -1, None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mj_writer = cv2.VideoWriter(str(out / 'chopsticks.mp4'), fourcc, fps, (width, height))
    combo_writer = (cv2.VideoWriter(str(out / 'chopsticks_side_by_side.mp4'), fourcc, fps, (width * 2, height))
                    if rgb_cap else None)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for index, frame, labels, ends in results:
            if ends and not args.realsense_view:  # smoothly follow the midpoint of the chopsticks
                center = np.mean([to_mj(q) for pair in ends.values() for q in pair], axis=0)
                cam.lookat[:] += 0.15 * (center - cam.lookat)
            renderer.update_scene(data, camera=cam)
            markers = [(labels[i][0] if i in labels else None, pt) for i, pt in frame['points'].items()]
            draw_chopsticks(renderer.scene, ends, markers)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            label_ends(img, renderer.scene, ends)
            overlay_text(img, index, ends)
            mj_writer.write(img)
            if combo_writer is not None:
                target = rgb_frames.get(index, rgb_pos)
                while rgb_pos < target:
                    ok, nxt = rgb_cap.read()
                    if not ok:
                        break
                    rgb_img, rgb_pos = nxt, rgb_pos + 1
                left = (cv2.resize(rgb_img, (width, height)) if rgb_img is not None
                        else np.zeros((height, width, 3), np.uint8))
                combo_writer.write(np.hstack((left, img)))
    finally:
        mj_writer.release()
        if combo_writer is not None:
            combo_writer.release()
        if rgb_cap is not None:
            rgb_cap.release()
        renderer.close()

    # 3) Summary.
    tracked = {n: sum(n in e for *_, e in results) for n in CHOPSTICKS}
    print(f'{len(results)} frames from {session}')
    for n, c in tracked.items():
        print(f'  {n}: ends estimated in {c}/{len(results)} frames')
    print('  Vega marker_id -> chopstick (marker distance from tip):')
    for i, votes in sorted(id_votes.items()):
        (name, k), c = votes.most_common(1)[0]
        print(f'    id {i:3d}: {name} @ {CHOPSTICKS[name]["markers"][k]:.0f} mm  ({c} frames)')
    print('Wrote:', ', '.join(str(out / f) for f in ('marker_labels.csv', 'chopstick_ends.csv', 'chopsticks.mp4')
                               + (('chopsticks_side_by_side.mp4',) if combo_writer else ())))


if __name__ == '__main__':
    main()
