import bpy
import json
import math
import os
import sys
import addon_utils
from mathutils import Vector, Matrix


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1:]
    out = {}
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            out["config"] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def emit_progress(p: float):
    p = max(0.0, min(1.0, float(p)))
    print(f"###PROGRESS {p:.6f}", flush=True)


def sanitize(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)
    return safe[:120] if safe else "unnamed"


def unique_name(desired: str, used: set[str]) -> str:
    if desired not in used:
        used.add(desired)
        return desired
    k = 1
    while True:
        cand = f"{desired}__{k:02d}"
        if cand not in used:
            used.add(cand)
            return cand
        k += 1


def enable_gltf_addon():
    try:
        addon_utils.enable("io_scene_gltf2", default_set=True)
    except Exception:
        pass


def import_glb(path: str):
    bpy.ops.import_scene.gltf(filepath=path)


def clear_imported_lights_and_cameras():
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def select_only(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def bbox_center_world(obj) -> Vector:
    mat = obj.matrix_world
    bb = [mat @ Vector(c) for c in obj.bound_box]
    c = Vector((0.0, 0.0, 0.0))
    for v in bb:
        c += v
    return c / max(1, len(bb))


def split_meshes(split_mode: str):
    split_mode = (split_mode or "AS_IS").upper()
    if split_mode == "AS_IS":
        return

    initial_mesh_objs = [o for o in list(bpy.data.objects) if o.type == "MESH"]
    used_names = set(o.name for o in bpy.data.objects)

    for obj in initial_mesh_objs:
        if obj.type != "MESH":
            continue

        base = obj.name
        if split_mode == "MATERIALS" and len(obj.material_slots) <= 1:
            continue

        before = set(o.name for o in bpy.data.objects)

        select_only(obj)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            if split_mode == "MATERIALS":
                bpy.ops.mesh.separate(type="MATERIAL")
            elif split_mode == "LOOSE_PARTS":
                bpy.ops.mesh.separate(type="LOOSE")
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            continue

        after = set(o.name for o in bpy.data.objects)
        created_names = list(after - before)

        derived = []
        for n in created_names:
            o = bpy.data.objects.get(n)
            if o and o.type == "MESH":
                derived.append(o)
        if obj and obj.name in bpy.data.objects and obj.type == "MESH":
            derived.append(obj)

        if not derived:
            continue

        if split_mode == "MATERIALS":
            for d in derived:
                mats = [ms.material.name if ms.material else "NoMat" for ms in d.material_slots]
                key = sanitize(mats[0] if mats else "NoMat")
                desired = f"{base}__MAT__{key}"
                d.name = unique_name(desired, used_names)

        elif split_mode == "LOOSE_PARTS":
            derived.sort(key=lambda o: (
                round(bbox_center_world(o).x, 4),
                round(bbox_center_world(o).y, 4),
                round(bbox_center_world(o).z, 4),
                o.name
            ))
            for idx, d in enumerate(derived):
                desired = f"{base}__PART__{idx:03d}"
                d.name = unique_name(desired, used_names)


def ensure_camera(scene: bpy.types.Scene):
    cam_data = bpy.data.cameras.new("SpriteCam")
    cam_obj = bpy.data.objects.new("SpriteCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def setup_world(scene: bpy.types.Scene, strength: float):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    w = scene.world
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = float(strength)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_sun(scene: bpy.types.Scene, name: str, energy: float, rot_deg, shadow: bool, angle_deg: float):
    light = bpy.data.lights.new(name, type="SUN")
    light.energy = float(energy)
    try:
        light.use_shadow = bool(shadow)
    except Exception:
        pass
    try:
        light.angle = math.radians(float(angle_deg))
    except Exception:
        pass

    obj = bpy.data.objects.new(name, light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (
        math.radians(rot_deg[0]),
        math.radians(rot_deg[1]),
        math.radians(rot_deg[2]),
    )
    return obj


def setup_lighting(scene: bpy.types.Scene, lighting_cfg: dict):
    preset = (lighting_cfg.get("preset", "STUDIO") or "STUDIO").upper()
    strength = float(lighting_cfg.get("strength", 1.0))
    world_strength = float(lighting_cfg.get("world_strength", 0.35))

    setup_world(scene, world_strength)

    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    if preset == "FLAT":
        add_sun(scene, "Key", 2.4 * strength, (55, 0, 45), shadow=False, angle_deg=12)
        add_sun(scene, "Fill", 2.4 * strength, (55, 0, 225), shadow=False, angle_deg=12)
        add_sun(scene, "Top", 1.2 * strength, (0, 0, 0), shadow=False, angle_deg=18)

    elif preset == "DRAMATIC":
        setup_world(scene, min(world_strength, 0.15))
        add_sun(scene, "Key", 5.0 * strength, (55, 0, 35), shadow=True, angle_deg=3)
        add_sun(scene, "Fill", 1.2 * strength, (65, 0, 230), shadow=False, angle_deg=10)
        add_sun(scene, "Rim", 2.2 * strength, (30, 0, 155), shadow=True, angle_deg=6)

    else:
        add_sun(scene, "Key", 3.6 * strength, (55, 0, 45), shadow=True, angle_deg=5)
        add_sun(scene, "Fill", 2.6 * strength, (65, 0, 225), shadow=False, angle_deg=12)
        add_sun(scene, "Rim", 1.6 * strength, (35, 0, 135), shadow=True, angle_deg=8)
        add_sun(scene, "Top", 1.0 * strength, (0, 0, 0), shadow=False, angle_deg=18)


def look_at(cam_obj, target: Vector):
    direction = target - cam_obj.location
    if direction.length < 1e-6:
        return
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()


def set_render_settings(scene, cfg):
    engine = cfg["engine"].upper().strip()
    if engine == "EEVEE":
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "BLENDER_EEVEE"
        try:
            scene.eevee.use_soft_shadows = True
        except Exception:
            pass
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(cfg.get("cycles_samples", 64))

    w, h = cfg["resolution"]
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100

    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"


# ---------- animation (exclusive) ----------
def iter_animdata_blocks():
    for obj in bpy.data.objects:
        ad = getattr(obj, "animation_data", None)
        if ad:
            yield ad
    for sk in bpy.data.shape_keys:
        ad = getattr(sk, "animation_data", None)
        if ad:
            yield ad


def apply_animation_exclusive(anim_name: str):
    for ad in iter_animdata_blocks():
        try:
            ad.action = None
        except Exception:
            pass

    if not anim_name or anim_name == "(static)":
        for ad in iter_animdata_blocks():
            for tr in getattr(ad, "nla_tracks", []) or []:
                tr.mute = True
                try:
                    tr.is_solo = False
                except Exception:
                    pass
                for st in tr.strips:
                    st.mute = True
        return None

    found = False
    min_fs = None
    max_fe = None

    for ad in iter_animdata_blocks():
        tracks = getattr(ad, "nla_tracks", None)
        if not tracks:
            continue

        for tr in tracks:
            is_target = (tr.name == anim_name)
            tr.mute = (not is_target)
            try:
                tr.is_solo = is_target
            except Exception:
                pass

            for st in tr.strips:
                st.mute = (not is_target)
                if is_target:
                    try:
                        st.blend_type = "REPLACE"
                    except Exception:
                        pass
                    try:
                        st.influence = 1.0
                    except Exception:
                        pass
                    try:
                        st.use_auto_blend = False
                    except Exception:
                        pass

            if is_target and tr.strips:
                found = True
                fs = min(float(s.frame_start) for s in tr.strips)
                fe = max(float(s.frame_end) for s in tr.strips)
                min_fs = fs if min_fs is None else min(min_fs, fs)
                max_fe = fe if max_fe is None else max(max_fe, fe)

    if found:
        return {"name": anim_name, "frame_start": min_fs, "frame_end": max_fe, "source": "NLA_EXCLUSIVE"}
    return None


# ---------- holdout / visibility ----------
def create_holdout_material():
    mat = bpy.data.materials.new("HOLDOUT_MAT")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hold = nt.nodes.new("ShaderNodeHoldout")
    nt.links.new(hold.outputs["Holdout"], out.inputs["Surface"])
    return mat


def set_object_holdout(obj, state: bool):
    try:
        if hasattr(obj, "is_holdout"):
            obj.is_holdout = state
            return True
    except Exception:
        pass
    try:
        cyc = getattr(obj, "cycles", None)
        if cyc and hasattr(cyc, "is_holdout"):
            cyc.is_holdout = state
            return True
    except Exception:
        pass
    return False


def apply_pass_state(mesh_objects, visible_set, holdout_set, holdout_mat, original_mats):
    for obj in mesh_objects:
        name = obj.name

        if name in visible_set:
            obj.hide_render = False
            set_object_holdout(obj, False)
            mats = original_mats.get(name, [])
            for slot, mat in zip(obj.material_slots, mats):
                slot.material = mat

        elif name in holdout_set:
            obj.hide_render = False
            ok = set_object_holdout(obj, True)
            if not ok:
                for slot in obj.material_slots:
                    slot.material = holdout_mat

        else:
            obj.hide_render = True
            set_object_holdout(obj, False)
            mats = original_mats.get(name, [])
            for slot, mat in zip(obj.material_slots, mats):
                slot.material = mat


# ---------- camera fit helpers ----------
def compute_bounds_eval(objects, depsgraph):
    inf = 1e18
    mins = Vector((inf, inf, inf))
    maxs = Vector((-inf, -inf, -inf))
    any_ok = False

    for obj in objects:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        bb = getattr(obj_eval, "bound_box", None)
        if not bb:
            continue
        mat = obj_eval.matrix_world
        for v in bb:
            w = mat @ Vector(v)
            mins.x = min(mins.x, w.x)
            mins.y = min(mins.y, w.y)
            mins.z = min(mins.z, w.z)
            maxs.x = max(maxs.x, w.x)
            maxs.y = max(maxs.y, w.y)
            maxs.z = max(maxs.z, w.z)
        any_ok = True

    if not any_ok:
        return None
    return mins, maxs


def find_main_armature(mesh_objects):
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        return None

    best = arms[0]
    best_score = -1
    for arm in arms:
        score = 0
        for m in mesh_objects:
            for mod in m.modifiers:
                if mod.type == "ARMATURE" and getattr(mod, "object", None) == arm:
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best = arm
    return best


def find_tracking_bone(arm_obj):
    if not arm_obj or not getattr(arm_obj, "pose", None):
        return None
    pbones = list(arm_obj.pose.bones)
    if not pbones:
        return None

    prefer = ["root", "hips", "pelvis"]
    for key in prefer:
        for pb in pbones:
            nm = pb.name.lower()
            if nm == key or key in nm:
                return pb

    roots = [pb for pb in pbones if pb.parent is None]
    if roots:
        roots.sort(key=lambda pb: len(pb.children), reverse=True)
        return roots[0]

    return pbones[0]


def basis_from_dir(dir_vec: Vector, roll_rad: float):
    dir_n = dir_vec.normalized()
    forward = -dir_n

    up0 = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(up0)) > 0.999:
        up0 = Vector((0.0, 1.0, 0.0))

    right = up0.cross(forward).normalized()
    up = forward.cross(right).normalized()

    if abs(roll_rad) > 1e-8:
        R = Matrix.Rotation(roll_rad, 3, forward)
        right = R @ right
        up = R @ up

    return dir_n, right, up


def required_dist_for_bounds(target: Vector, bounds, dir_n: Vector, right: Vector, up: Vector, tanx: float, tany: float):
    mins, maxs = bounds
    req = 0.1

    for x in (mins.x, maxs.x):
        for y in (mins.y, maxs.y):
            for z in (mins.z, maxs.z):
                corner = Vector((x, y, z))
                p_rel = corner - target

                # component toward camera location direction
                along = p_rel.dot(dir_n)

                # screen-space half extents around target
                hx = abs(p_rel.dot(right))
                hy = abs(p_rel.dot(up))

                need = along + max(hx / max(tanx, 1e-6), hy / max(tany, 1e-6), 1e-6)
                if need > req:
                    req = need

    return req


def required_ortho_scale_for_bounds(target: Vector, bounds, dir_n: Vector, right: Vector, up: Vector, aspect: float):
    mins, maxs = bounds
    max_x = 0.0
    max_y = 0.0

    for x in (mins.x, maxs.x):
        for y in (mins.y, maxs.y):
            for z in (mins.z, maxs.z):
                corner = Vector((x, y, z))
                p_rel = corner - target
                max_x = max(max_x, abs(p_rel.dot(right)))
                max_y = max(max_y, abs(p_rel.dot(up)))

    # Blender ortho_scale ≈ width of view. Height = width / aspect.
    need_w = 2.0 * max_x
    need_w_for_h = 2.0 * max_y * max(aspect, 1e-6)
    return max(need_w, need_w_for_h, 0.001)


def precompute_targets_and_bounds(scene, frames, ref_mesh_objects, depsgraph, arm_obj, root_pb, target_offset_v: Vector, progress_base: float, progress_span: float):
    targets = []
    bounds_list = []

    total = max(1, len(frames))
    for i, frame in enumerate(frames):
        scene.frame_set(int(frame))
        depsgraph.update()

        b = compute_bounds_eval(ref_mesh_objects, depsgraph)
        bounds_list.append(b)

        if arm_obj and root_pb:
            t = arm_obj.matrix_world @ root_pb.head
        else:
            if b:
                mins, maxs = b
                t = (mins + maxs) * 0.5
            else:
                t = Vector((0.0, 0.0, 0.0))

        targets.append(t + target_offset_v)

        if i == 0 or i == total - 1 or (i % max(1, total // 30) == 0):
            emit_progress(progress_base + progress_span * ((i + 1) / total))

    return targets, bounds_list


def main():
    args = parse_args()
    cfg_path = args.get("config")
    if not cfg_path:
        print("No --config provided", file=sys.stderr)
        sys.exit(2)

    cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    input_glb = cfg["input_glb"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_gltf_addon()
    import_glb(input_glb)

    clear_imported_lights_and_cameras()

    split_mode = cfg.get("import", {}).get("split_mode", "AS_IS")
    split_meshes(split_mode)

    scene = bpy.context.scene
    set_render_settings(scene, cfg)
    setup_lighting(scene, cfg.get("lighting", {}))

    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]

    cam_obj = ensure_camera(scene)
    cam = cam_obj.data

    camera_cfg = cfg.get("camera", {})
    fov_deg = float(camera_cfg.get("fov_deg", 50.0))
    try:
        cam.angle = math.radians(fov_deg)
    except Exception:
        pass

    roll_deg = float(camera_cfg.get("roll_deg", 0.0))
    roll_rad = math.radians(roll_deg)

    dist_mult = float(camera_cfg.get("distance_mult", 1.25))
    # Можно увеличить если нужно ещё безопаснее: 1.10 -> 1.20 / 1.30
    fit_margin = float(camera_cfg.get("fit_margin", 1.15))

    use_ortho = bool(camera_cfg.get("ortho", False))
    cam.type = "ORTHO" if use_ortho else "PERSP"

    target_offset = camera_cfg.get("target_offset", [0.0, 0.0, 0.0])
    target_offset_v = Vector((float(target_offset[0]), float(target_offset[1]), float(target_offset[2])))

    # Animation
    anim_cfg = cfg["animation"]
    scene.render.fps = int(anim_cfg.get("fps", 24))
    step = max(1, int(anim_cfg.get("step", 1)))

    anim_name = anim_cfg.get("name", "(static)")
    anim_info = apply_animation_exclusive(anim_name)

    if anim_info:
        fs = int(round(anim_info["frame_start"]))
        fe = int(round(anim_info["frame_end"]))
        if fe < fs:
            fs, fe = fe, fs
    else:
        fs, fe = 0, 0

    frames = list(range(fs, fe + 1, step))
    if not frames:
        frames = [0]

    # Angles + layers
    angles = cfg["angles"]
    layers = cfg["layers"]
    export_cfg = cfg["export"]

    ordered = sorted(layers, key=lambda x: int(x.get("order", 0)))
    modes = {x["name"]: x.get("mode", "VISIBLE") for x in ordered}
    order_names = [x["name"] for x in ordered]

    # Reference meshes for camera fitting: all VISIBLE meshes
    ref_names = [n for n in order_names if modes.get(n) == "VISIBLE"]
    if ref_names:
        ref_set = set(ref_names)
        ref_mesh_objects = [o for o in mesh_objects if o.name in ref_set]
    else:
        ref_mesh_objects = list(mesh_objects)

    # Armature root tracking (stable pivot, no jitter)
    arm_obj = find_main_armature(mesh_objects)
    root_pb = find_tracking_bone(arm_obj) if arm_obj else None
    if arm_obj and root_pb:
        print(f"INFO: camera pivot = root bone: {arm_obj.name} / {root_pb.name}", flush=True)
    else:
        print("INFO: camera pivot = bounds center (no armature/root)", flush=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # 1) Precompute target + bounds per frame (fast)
    pre_w = 0.10
    emit_progress(0.0)
    print("INFO: precomputing targets + bounds...", flush=True)
    targets, bounds_list = precompute_targets_and_bounds(
        scene, frames, ref_mesh_objects, depsgraph, arm_obj, root_pb, target_offset_v,
        progress_base=0.0, progress_span=pre_w
    )

    res_x = int(scene.render.resolution_x)
    res_y = int(scene.render.resolution_y)
    aspect = (float(res_x) / float(res_y)) if res_y else 1.0

    # 2) Compute ONE safe distance / ortho_scale that fits:
    #    - all frames (with step)
    #    - all angles
    print("INFO: computing safe camera fit (no clipping)...", flush=True)

    dirs = []
    for ang in angles:
        yaw = math.radians(float(ang["yaw"]))
        pitch = math.radians(float(ang["pitch"]))
        dir_n = Vector((
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ))
        if dir_n.length < 1e-6:
            dir_n = Vector((0.0, 1.0, 0.0))
        dirs.append(dir_n.normalized())

    if cam.type == "ORTHO":
        max_needed_scale = 0.001
        for dir_n in dirs:
            dir_n, right, up = basis_from_dir(dir_n, roll_rad)
            needed_scale_dir = 0.001
            for i in range(len(frames)):
                b = bounds_list[i]
                if not b:
                    continue
                needed_scale_dir = max(
                    needed_scale_dir,
                    required_ortho_scale_for_bounds(targets[i], b, dir_n, right, up, aspect)
                )
            max_needed_scale = max(max_needed_scale, needed_scale_dir)

        req_scale = max_needed_scale * fit_margin * dist_mult
        user_scale = float(camera_cfg.get("ortho_scale", 0.0))
        cam.ortho_scale = max(user_scale, req_scale)

        # distance for positioning only
        dist = max(1.0, cam.ortho_scale * 2.0)

    else:
        try:
            tanx = math.tan(float(cam.angle_x) / 2.0)
            tany = math.tan(float(cam.angle_y) / 2.0)
        except Exception:
            tanx = math.tan(math.radians(fov_deg) / 2.0)
            tany = tanx

        max_needed_d = 0.1
        for dir_n in dirs:
            dir_n, right, up = basis_from_dir(dir_n, roll_rad)
            needed_d_dir = 0.1
            for i in range(len(frames)):
                b = bounds_list[i]
                if not b:
                    continue
                needed_d_dir = max(
                    needed_d_dir,
                    required_dist_for_bounds(targets[i], b, dir_n, right, up, tanx, tany)
                )
            max_needed_d = max(max_needed_d, needed_d_dir)

        dist = max_needed_d * fit_margin * dist_mult
        dist = max(dist, 0.1)

    # 3) Passes
    passes = []
    if export_cfg.get("combined", True):
        vis = [n for n in order_names if modes.get(n) == "VISIBLE"]
        hol = [n for n in order_names if modes.get(n) == "MASK"]
        passes.append({"name": "combined", "visible": vis, "holdout": hol})

    if export_cfg.get("per_layer", True):
        for i, name in enumerate(order_names):
            if modes.get(name) != "VISIBLE":
                continue
            above = order_names[i + 1:]
            holdout = [n for n in above if modes.get(n) in ("VISIBLE", "MASK")]
            passes.append({"name": f"layer_{sanitize(name)}", "visible": [name], "holdout": holdout})

    holdout_mat = create_holdout_material()
    original_mats = {o.name: [s.material for s in o.material_slots] for o in mesh_objects}

    total_renders = max(1, len(passes) * len(angles) * len(frames))
    done = 0

    manifest = {
        "input_glb": input_glb,
        "output_dir": out_dir,
        "frame_size": cfg["resolution"],
        "fps": int(scene.render.fps),
        "angles": angles,
        "frames": frames,
        "animation": anim_info or {"name": "(static)", "frame_start": 0, "frame_end": 0, "source": "NONE"},
        "passes": [],
        "camera": camera_cfg,
        "lighting": cfg.get("lighting", {}),
        "import": {"split_mode": split_mode},
        "camera_fit": {
            "fit_margin": float(fit_margin),
            "distance_mult": float(dist_mult),
            "mode": "ORTHO" if cam.type == "ORTHO" else "PERSP",
            "dist": float(dist),
            "ortho_scale": float(cam.ortho_scale) if cam.type == "ORTHO" else None,
        },
    }

    # 4) Render
    for p in passes:
        pass_dir = os.path.join(out_dir, p["name"])
        os.makedirs(pass_dir, exist_ok=True)
        manifest["passes"].append({"name": p["name"], "dir": p["name"]})

        apply_pass_state(mesh_objects, set(p["visible"]), set(p["holdout"]), holdout_mat, original_mats)

        for ai, ang in enumerate(angles):
            a_dir = os.path.join(pass_dir, f"angle_{ai:02d}")
            os.makedirs(a_dir, exist_ok=True)

            yaw = math.radians(float(ang["yaw"]))
            pitch = math.radians(float(ang["pitch"]))

            dir_n = Vector((
                math.sin(yaw) * math.cos(pitch),
                math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            ))
            if dir_n.length < 1e-6:
                dir_n = Vector((0.0, 1.0, 0.0))
            dir_n.normalize()

            dir_vec = dir_n * dist

            for fi, frame in enumerate(frames):
                scene.frame_set(int(frame))
                target = targets[fi]

                cam_obj.location = target + dir_vec
                look_at(cam_obj, target)
                cam_obj.rotation_euler.rotate_axis("Z", roll_rad)

                scene.render.filepath = os.path.join(a_dir, f"frame_{fi:04d}.png")
                bpy.ops.render.render(write_still=True)

                done += 1
                if done == 1 or done == total_renders or (done % max(1, total_renders // 200) == 0):
                    prog = pre_w + (1.0 - pre_w) * (done / total_renders)
                    emit_progress(prog)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    emit_progress(1.0)
    print(f"###MANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
