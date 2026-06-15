from __future__ import annotations

STANDARD_RS_WIDTH = 848
STANDARD_RS_HEIGHT = 480
STANDARD_RS_FPS = 30

STANDARD_COLOR_STREAM = "color"
STANDARD_COLOR_FORMAT = "bgr8"
STANDARD_DEPTH_STREAM = "depth"
STANDARD_DEPTH_FORMAT = "z16"

DEPTH_ALIGNED_TO = "color"
DEPTH_PNG_DTYPE = "uint16"
DEPTH_PNG_UNIT = "millimeter"
DEPTH_PNG_UNIT_M = 0.001

CAMERA_METADATA_FILE = "camera_metadata.json"


def standard_realsense_profile() -> dict:
    return {
        "width": STANDARD_RS_WIDTH,
        "height": STANDARD_RS_HEIGHT,
        "fps": STANDARD_RS_FPS,
        "color_stream": STANDARD_COLOR_STREAM,
        "color_format": STANDARD_COLOR_FORMAT,
        "depth_stream": STANDARD_DEPTH_STREAM,
        "depth_format": STANDARD_DEPTH_FORMAT,
        "depth_aligned_to": DEPTH_ALIGNED_TO,
        "depth_png_dtype": DEPTH_PNG_DTYPE,
        "depth_png_unit": DEPTH_PNG_UNIT,
        "depth_png_unit_m": DEPTH_PNG_UNIT_M,
    }
