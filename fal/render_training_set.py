"""Render a LoRA training set from the DASH rig, headless in Blender.

Salvaged footage is the wrong shape for a subject LoRA and we proved it the
hard way: the visualizers hold one static shot for minutes, the episodes cut
at a 0.2 s median with the character absent from most segments, and the
promos are vertical. Curating all of it yielded thirteen usable clips against
a twenty-to-fifty recommendation.

The rig has none of those problems. It gives canonical geometry, total
control of angle and light, native 16:9 at 24 fps, no captions, no
compression artefacts, and as many clips as there is patience for — which
makes synthetic data from the source model the best case here, not a
compromise.

Run it (Blender 4.5 LTS; 5.x has no Intel macOS build):

    blender --background "Dash - rigged.blend" \
        --python fal/render_training_set.py -- --out fal/clips_rig

Then train on the result:

    python fal/train_lora.py fal/clips_rig --trigger DASH250

Every clip length here satisfies the trainer's `frames % 17 == 5` rule, so
nothing needs `auto_scale_input`. The script builds its own camera and key
lights rather than trusting whatever the file ships with, so a scene saved
mid-edit still renders a consistent set.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# The trainer's legal clip lengths (n % 17 == 5) at 24 fps: 73 frames is
# 3.04 s, 124 is 5.17 s. Short clips buy variety per render-hour; the longer
# ones carry motion the short ones cut off.
SHORT, LONG = 73, 124
WIDTH, HEIGHT, FPS = 1024, 576, 24

# Three lighting states, as (key energy, key colour, rim colour). The LoRA
# should learn DASH's design, not one studio setup — so the set spans the
# neon world he actually lives in.
LIGHTING = {
    "neutral": (900.0, (1.00, 0.98, 0.95), (0.85, 0.90, 1.00)),
    "neon":    (700.0, (0.20, 0.85, 1.00), (1.00, 0.35, 0.80)),
    "warm":    (800.0, (1.00, 0.72, 0.42), (0.35, 0.55, 1.00)),
}


def character_bounds() -> tuple[Vector, float]:
    """World-space centre and radius of every visible mesh in the file."""
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, world))
            hi = Vector(map(max, hi, world))
            found = True
    if not found:
        raise SystemExit("no visible meshes in the file — is the rig hidden?")
    centre = (lo + hi) / 2.0
    return centre, max((hi - lo).length / 2.0, 0.001)


def build_rig_camera(centre: Vector, radius: float):
    """A camera that orbits the character, independent of the file's own."""
    data = bpy.data.cameras.new("trainset_cam")
    data.lens = 50.0
    cam = bpy.data.objects.new("trainset_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    pivot = bpy.data.objects.new("trainset_pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = centre
    cam.parent = pivot

    track = cam.constraints.new("TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return cam, pivot


def build_lights(centre: Vector, radius: float, state: str) -> list:
    """Key plus rim, sized to the subject. Returns objects to clean up."""
    energy, key_colour, rim_colour = LIGHTING[state]
    made = []
    for name, offset, colour, scale in (
        ("key", Vector((1.2, -1.6, 1.1)), key_colour, 1.0),
        ("rim", Vector((-1.4, 1.2, 0.8)), rim_colour, 0.6),
    ):
        data = bpy.data.lights.new(f"trainset_{name}", type="AREA")
        data.energy = energy * scale * (radius ** 2)
        data.color = colour
        data.size = radius * 1.5
        light = bpy.data.objects.new(f"trainset_{name}", data)
        light.location = centre + offset * radius * 2.5
        constraint = light.constraints.new("TRACK_TO")
        empty = bpy.data.objects.new(f"trainset_{name}_aim", None)
        empty.location = centre
        bpy.context.scene.collection.objects.link(empty)
        constraint.target = empty
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        bpy.context.scene.collection.objects.link(light)
        made += [light, empty]
    return made


def frame_subject(cam, pivot, centre: Vector, radius: float,
                  distance: float, height: float, yaw: float) -> None:
    """Place the orbit rig for one shot. `distance` is in subject radii."""
    pivot.location = centre + Vector((0.0, 0.0, radius * height))
    pivot.rotation_euler = (0.0, 0.0, math.radians(yaw))
    cam.location = Vector((0.0, -radius * distance, 0.0))


def configure_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(
        bpy.types, "SceneEEVEE") else "BLENDER_EEVEE"
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"


def shot_list(has_action: bool) -> list[dict]:
    """The coverage a subject LoRA actually generalises from.

    Turntables teach the silhouette from every side; the close-ups teach the
    face; the wardrobe passes teach the boots and denim that text alone kept
    getting wrong. Action shots only appear when the rig ships animation —
    without it their frames would be a static pose and would teach the LoRA
    that DASH does not move.
    """
    shots: list[dict] = []
    for state in ("neutral", "neon", "warm"):
        shots += [
            dict(name=f"turntable_full_{state}", light=state, frames=LONG,
                 distance=3.0, height=0.05, yaw=(0, 360)),
            dict(name=f"turntable_bust_{state}", light=state, frames=SHORT,
                 distance=1.5, height=0.62, yaw=(30, 210)),
            dict(name=f"face_{state}", light=state, frames=SHORT,
                 distance=0.9, height=0.80, yaw=(-25, 25)),
            dict(name=f"boots_{state}", light=state, frames=SHORT,
                 distance=1.1, height=-0.55, yaw=(-40, 60)),
            dict(name=f"low_hero_{state}", light=state, frames=SHORT,
                 distance=2.2, height=-0.25, yaw=(200, 340)),
        ]
    if has_action:
        for state in ("neutral", "neon"):
            shots.append(dict(name=f"action_{state}", light=state, frames=LONG,
                              distance=2.6, height=0.15, yaw=(45, 75),
                              use_action=True))
    return shots


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("fal/clips_rig"))
    parser.add_argument("--only", default=None,
                        help="substring filter, for re-rendering one shot")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    configure_scene()
    centre, radius = character_bounds()
    cam, pivot = build_rig_camera(centre, radius)

    actions = list(bpy.data.actions)
    rigged = [o for o in bpy.data.objects
              if o.type == "ARMATURE" and o.animation_data and o.animation_data.action]
    has_action = bool(actions and rigged)
    print(f"[trainset] subject radius {radius:.2f}, "
          f"{len(actions)} action(s), {len(rigged)} animated armature(s)")

    shots = [s for s in shot_list(has_action)
             if not args.only or args.only in s["name"]]
    scene = bpy.context.scene

    for index, shot in enumerate(shots, 1):
        lights = build_lights(centre, radius, shot["light"])
        frames = shot["frames"]
        start_yaw, end_yaw = shot["yaw"]

        # A shot either plays the rig's own animation or orbits a still pose;
        # either way the camera sweeps, so no clip is a frozen frame.
        if shot.get("use_action") and has_action:
            action = rigged[0].animation_data.action
            first = int(action.frame_range[0])
            scene.frame_start = first
            scene.frame_end = first + frames - 1
        else:
            scene.frame_start = 1
            scene.frame_end = frames

        for offset in range(frames):
            progress = offset / max(frames - 1, 1)
            frame_subject(cam, pivot, centre, radius, shot["distance"],
                          shot["height"], start_yaw + (end_yaw - start_yaw) * progress)
            pivot.keyframe_insert("rotation_euler", frame=scene.frame_start + offset)
            pivot.keyframe_insert("location", frame=scene.frame_start + offset)
            cam.keyframe_insert("location", frame=scene.frame_start + offset)

        scene.render.filepath = str(args.out / shot["name"])
        print(f"[trainset] {index}/{len(shots)} {shot['name']}: "
              f"{frames} frames ({frames / FPS:.2f}s)")
        bpy.ops.render.render(animation=True)

        pivot.animation_data_clear()
        cam.animation_data_clear()
        for obj in lights:
            bpy.data.objects.remove(obj, do_unlink=True)

    print(f"[trainset] wrote {len(shots)} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
