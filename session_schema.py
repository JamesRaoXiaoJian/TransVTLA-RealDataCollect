from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


WORLD_CAMERA = "world_camera"
WRIST_CAMERA = "wrist_camera"
RGB_DIR = "rgb"
DEPTH_DIR = "depth"

LEGACY_WORLD_RGB = "dji"
LEGACY_WRIST_RGB = "realsense_rgb"
LEGACY_WRIST_DEPTH = "realsense_depth"

RGB_EXT = ".jpg"
DEPTH_EXT = ".png"


@dataclass(frozen=True)
class CameraDirs:
    name: str
    rgb: Path
    depth: Optional[Path]


@dataclass(frozen=True)
class SessionLayout:
    root: Path
    world: CameraDirs
    wrist: CameraDirs
    is_legacy: bool = False


def camera_root(session_root: Path, camera_name: str) -> Path:
    return session_root / camera_name


def camera_rgb_dir(session_root: Path, camera_name: str) -> Path:
    return camera_root(session_root, camera_name) / RGB_DIR


def camera_depth_dir(session_root: Path, camera_name: str) -> Path:
    return camera_root(session_root, camera_name) / DEPTH_DIR


def is_dual_realsense_session(session_dir: Path) -> bool:
    return (
        camera_rgb_dir(session_dir, WORLD_CAMERA).is_dir()
        and camera_depth_dir(session_dir, WORLD_CAMERA).is_dir()
        and camera_rgb_dir(session_dir, WRIST_CAMERA).is_dir()
        and camera_depth_dir(session_dir, WRIST_CAMERA).is_dir()
    )


def is_legacy_realsense_dji_session(session_dir: Path) -> bool:
    return (session_dir / LEGACY_WORLD_RGB).is_dir() and (session_dir / LEGACY_WRIST_RGB).is_dir()


def resolve_session_layout(session_dir: Path) -> SessionLayout:
    if is_dual_realsense_session(session_dir):
        return SessionLayout(
            root=session_dir,
            world=CameraDirs(
                name=WORLD_CAMERA,
                rgb=camera_rgb_dir(session_dir, WORLD_CAMERA),
                depth=camera_depth_dir(session_dir, WORLD_CAMERA),
            ),
            wrist=CameraDirs(
                name=WRIST_CAMERA,
                rgb=camera_rgb_dir(session_dir, WRIST_CAMERA),
                depth=camera_depth_dir(session_dir, WRIST_CAMERA),
            ),
            is_legacy=False,
        )

    if is_legacy_realsense_dji_session(session_dir):
        wrist_depth = session_dir / LEGACY_WRIST_DEPTH
        return SessionLayout(
            root=session_dir,
            world=CameraDirs(name=WORLD_CAMERA, rgb=session_dir / LEGACY_WORLD_RGB, depth=None),
            wrist=CameraDirs(
                name=WRIST_CAMERA,
                rgb=session_dir / LEGACY_WRIST_RGB,
                depth=wrist_depth if wrist_depth.is_dir() else None,
            ),
            is_legacy=True,
        )

    raise FileNotFoundError(
        "Session missing dual RealSense layout "
        "(world_camera/{rgb,depth}, wrist_camera/{rgb,depth}) "
        "or legacy layout (dji/, realsense_rgb/)."
    )


def find_session_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []

    candidates: set[Path] = set()
    if base.is_dir():
        candidates.add(base)

    for world_dir in base.rglob(WORLD_CAMERA):
        candidates.add(world_dir.parent)
    for dji_dir in base.rglob(LEGACY_WORLD_RGB):
        candidates.add(dji_dir.parent)

    sessions = []
    for candidate in candidates:
        try:
            resolve_session_layout(candidate)
        except FileNotFoundError:
            continue
        sessions.append(candidate)

    return sorted(set(sessions), key=lambda p: str(p))


def _stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.glob(f"*{suffix}")}


def common_frame_stems(layout: SessionLayout) -> list[str]:
    required_sets = [
        _stems(layout.world.rgb, RGB_EXT),
        _stems(layout.wrist.rgb, RGB_EXT),
    ]

    for depth_dir in (layout.world.depth, layout.wrist.depth):
        if depth_dir is not None:
            required_sets.append(_stems(depth_dir, DEPTH_EXT))

    if not required_sets or any(not stems for stems in required_sets):
        return []
    return sorted(set.intersection(*required_sets))


def rgb_path(camera: CameraDirs, stem: str) -> Path:
    return camera.rgb / f"{stem}{RGB_EXT}"


def depth_path(camera: CameraDirs, stem: str) -> Optional[Path]:
    if camera.depth is None:
        return None
    return camera.depth / f"{stem}{DEPTH_EXT}"
