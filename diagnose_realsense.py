"""Diagnostic script for RealSense camera connection issues."""

import sys

try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pyrealsense2: {e}")
    sys.exit(1)

# Check for connected devices
print("\nChecking for connected RealSense devices...")
context = rs.context()
devices = context.query_devices()

print(f"Number of devices found: {len(devices)}")

if len(devices) == 0:
    print("\n✗ No RealSense devices detected!")
    print("\nTroubleshooting steps:")
    print("1. Check physical USB connection")
    print("2. Try a different USB port (preferably USB 3.0/3.1)")
    print("3. Check Device Manager for unknown or error devices")
    print("4. Reinstall RealSense SDK drivers:")
    print("   - Download from: https://github.com/IntelRealSense/librealsense/releases")
    print("   - Run: pip install --upgrade pyrealsense2")
    print("5. On Windows, install: Intel RealSense D400 drivers from Device Manager")
    sys.exit(1)

print("\nConnected devices:")
for i, device in enumerate(devices):
    print(f"\n  Device {i}:")
    print(f"    Name: {device.get_info(rs.camera_info.name)}")
    print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    
    # Try to initialize the device
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        print(f"    ✓ Device initialized successfully")
        pipeline.stop()
    except Exception as e:
        print(f"    ✗ Failed to initialize: {e}")

print("\n✓ Diagnostic complete")
