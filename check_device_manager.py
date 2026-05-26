"""Check Windows Device Manager for RealSense devices."""

import subprocess
import re

print("Checking Windows Device Manager for RealSense devices...\n")

# Query all USB devices
result = subprocess.run(
    ["wmic", "logicaldisk", "get", "name"],
    capture_output=True,
    text=True
)

# Better approach: use Windows Registry to check for RealSense devices
try:
    result = subprocess.run(
        ["wmic", "path", "win32_pnpentity", "get", "name", "/format:list"],
        capture_output=True,
        text=True
    )
    
    devices = result.stdout.strip().split('\n')
    realsense_devices = [d for d in devices if 'realsense' in d.lower() or 'intel' in d.lower() or 'usb' in d.lower()]
    
    if realsense_devices:
        print("Found USB/RealSense devices in Device Manager:")
        for device in realsense_devices[:20]:  # Show first 20
            if device.strip():
                print(f"  • {device.strip()}")
    else:
        print("No RealSense devices found in Device Manager")
except Exception as e:
    print(f"Error querying devices: {e}")

print("\n" + "="*60)
print("Alternative: Check USB devices with detailed info")
print("="*60)

try:
    result = subprocess.run(
        ["powershell", "-Command", 
         "Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -like '*USB*' -or $_.Name -like '*Intel*' -or $_.Name -like '*Real*'} | Select-Object Name, Status, ConfigManagerErrorCode"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("Manual steps to verify:")
print("="*60)
print("1. Press: Win + R")
print("2. Type: devmgmt.msc")
print("3. Press: Enter")
print("4. Look for:")
print("   - Intel RealSense D400 series")
print("   - Intel RealSense Depth Camera")
print("   - Unknown devices (⚠)")
print("5. If found with ⚠, right-click → 'Update driver' → 'Browse my computer'")
print("6. Navigate to: C:\\Program Files\\Intel RealSense SDK 2.0\\drivers")
