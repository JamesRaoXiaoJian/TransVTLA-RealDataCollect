# PowerShell script to diagnose RealSense issues
Write-Host "RealSense Camera Troubleshooting Check" -ForegroundColor Cyan
Write-Host "=" * 60

# Check 1: SDK Installation
Write-Host "`n[1] Checking RealSense SDK Installation..." -ForegroundColor Yellow
$sdkPath = "C:\Users\james\Documents\RealSense SDK 2.0"
if (Test-Path $sdkPath) {
    Write-Host "✓ SDK installed at: $sdkPath" -ForegroundColor Green
    Get-ChildItem $sdkPath -Directory | ForEach-Object { Write-Host "  • $_" }
} else {
    Write-Host "✗ SDK not found" -ForegroundColor Red
    Write-Host "  Expected at: $sdkPath" -ForegroundColor Yellow
}

# Check 2: Drivers Path
Write-Host "`n[2] Checking RealSense Drivers..." -ForegroundColor Yellow
$driverPath = "$sdkPath\drivers"
if (Test-Path $driverPath) {
    Write-Host "✓ Drivers folder found" -ForegroundColor Green
    Get-ChildItem $driverPath | ForEach-Object { Write-Host "  • $_" }
} else {
    Write-Host "✗ Drivers folder not found" -ForegroundColor Red
}

# Check 3: pyrealsense2 Python module
Write-Host "`n[3] Checking pyrealsense2 Python module..." -ForegroundColor Yellow
$pythonExe = "c:\Users\james\TransVTLA-RealDataCollect\.venv\Scripts\python.exe"
$moduleTest = & $pythonExe -c "import pyrealsense2; print('OK')" 2>&1
if ($moduleTest -eq "OK") {
    Write-Host "✓ pyrealsense2 module is available" -ForegroundColor Green
} else {
    Write-Host "✗ pyrealsense2 module not working" -ForegroundColor Red
}

# Check 4: Critical troubleshooting steps
Write-Host "`n[4] Physical Connection Checklist:" -ForegroundColor Yellow
Write-Host "  ☐ USB cable securely connected to RealSense camera"
Write-Host "  ☐ Connected to USB 3.0/3.1 port (usually blue, not black)"
Write-Host "  ☐ Try different USB port on computer"
Write-Host "  ☐ Try different USB cable"

Write-Host "`n[5] Driver Installation Steps:" -ForegroundColor Yellow
Write-Host "  1. Open Device Manager (Win + R → devmgmt.msc → Enter)"
Write-Host "  2. Look for 'Intel RealSense' or devices with yellow ⚠ mark"
Write-Host "  3. Right-click problematic device → 'Update driver'"
Write-Host "  4. Select 'Browse my computer for driver software'"
Write-Host "  5. Browse to: $driverPath"
Write-Host "  6. Click 'Next' and wait for installation"

Write-Host "`n[6] After fixing hardware/drivers, re-run diagnostic:" -ForegroundColor Cyan
Write-Host "  & '$pythonExe' diagnose_realsense.py`n" -ForegroundColor Green
