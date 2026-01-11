import bpy
import json
import sys
import addon_utils
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1:]
    out = {"split_mode": "AS_IS"}
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            out["input"] = args[i + 1]
            i += 2
        elif args[i] == "--split_mode" and i + 1 < len(args):
            out["split_mode"] = args[i + 1].strip().upper()
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
    total = max(1, len(initial_mesh_objs))
    used_names = set(o.name for o in bpy.data.objects)

    print(f"INFO: split_mode={split_mode} mesh_objects={len(initial_mesh_objs)}", flush=True)

    # Начнём прогресс с 0.10 — импорт уже случился
    emit_progress(0.10)

    for idx, obj in enumerate(initial_mesh_objs):
        if obj.type != "MESH":
            continue

        base = obj.name

        # Визуальная “живость”
        emit_progress(0.10 + 0.75 * (idx / total))

        try:
            vcount = len(obj.data.vertices) if obj.data else 0
            pcount = len(obj.data.polygons) if obj.data else 0
        except Exception:
            vcount = 0
            pcount = 0

        print(f"INFO: splitting {base} verts={vcount} polys={pcount}", flush=True)
        if split_mode == "LOOSE_PARTS":
            print("INFO: (loose parts) this can be slow on heavy meshes...", flush=True)

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
        except Exception as e:
            print(f"WARN: split failed for {base}: {e}", flush=True)
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
            for i2, d in enumerate(derived):
                desired = f"{base}__PART__{i2:03d}"
                d.name = unique_name(desired, used_names)

        emit_progress(0.10 + 0.75 * ((idx + 1) / total))

    emit_progress(0.90)


def collect_mesh_details():
    details = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        try:
            verts = len(obj.data.vertices)
            polys = len(obj.data.polygons)
        except Exception:
            verts = 0
            polys = 0

        mats = []
        for ms in obj.material_slots:
            mats.append(ms.material.name if ms.material else "NoMat")

        details.append({
            "name": obj.name,
            "verts": int(verts),
            "polys": int(polys),
            "materials": mats,
        })

    details.sort(key=lambda d: d["name"].lower())
    return details


def _scan_animdata(ad, ranges: dict):
    tracks = getattr(ad, "nla_tracks", None)
    if not tracks:
        return
    for track in tracks:
        if not track.strips:
            continue
        name = track.name
        starts = [s.frame_start for s in track.strips]
        ends = [s.frame_end for s in track.strips]
        fs = float(min(starts))
        fe = float(max(ends))
        if name in ranges:
            ranges[name]["frame_start"] = min(ranges[name]["frame_start"], fs)
            ranges[name]["frame_end"] = max(ranges[name]["frame_end"], fe)
        else:
            ranges[name] = {"name": name, "frame_start": fs, "frame_end": fe}


def collect_animations():
    ranges = {}

    for obj in bpy.data.objects:
        ad = getattr(obj, "animation_data", None)
        if ad:
            _scan_animdata(ad, ranges)

    for sk in bpy.data.shape_keys:
        ad = getattr(sk, "animation_data", None)
        if ad:
            _scan_animdata(ad, ranges)

    if ranges:
        anims = list(ranges.values())
        anims.sort(key=lambda a: a["name"].lower())
        return anims

    anims = []
    for act in bpy.data.actions:
        fr = act.frame_range
        anims.append({"name": act.name, "frame_start": float(fr[0]), "frame_end": float(fr[1])})
    anims.sort(key=lambda a: a["name"].lower())
    return anims


def main():
    a = parse_args()
    path = a.get("input")
    if not path:
        print("No --input provided", file=sys.stderr)
        sys.exit(2)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_gltf_addon()

    emit_progress(0.02)
    import_glb(path)
    emit_progress(0.08)

    split_mode = a.get("split_mode", "AS_IS")
    split_meshes(split_mode)

    mesh_details = collect_mesh_details()
    meshes = [d["name"] for d in mesh_details]

    stats = {
        "objects_total": int(len(bpy.data.objects)),
        "mesh_objects": int(len([o for o in bpy.data.objects if o.type == "MESH"])),
        "armatures": int(len([o for o in bpy.data.objects if o.type == "ARMATURE"])),
        "materials": int(len(bpy.data.materials)),
        "images": int(len(bpy.data.images)),
        "split_mode": split_mode,
    }

    data = {
        "meshes": meshes,
        "mesh_details": mesh_details,
        "animations": collect_animations(),
        "stats": stats,
    }

    emit_progress(0.97)
    print("###BEGIN_JSON", flush=True)
    print(json.dumps(data, ensure_ascii=False), flush=True)
    print("###END_JSON", flush=True)
    emit_progress(1.0)


if __name__ == "__main__":
    main()
