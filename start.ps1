# Optimized PPG Service Launcher
# Starts all services in correct dependency order with health checks
# Version 2.0 - Tested and Working

param(
    [switch]$SkipNgrok = $false
)

$ErrorActionPreference = "Continue"
$projectPath = "C:\Users\nazeh\BioInfo Trials\3x1-PPG\Dev\3x1-PPG"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PPG Monitoring System Launcher v2.0  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if port is in use
function Test-ServicePort {
    param([int]$Port)
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return $null -ne $connection
    } catch {
        return $false
    }
}

# Function to wait for service to be ready
function Wait-ForServiceReady {
    param(
        [int]$Port,
        [string]$ServiceName,
        [int]$TimeoutSeconds = 30
    )
    
    Write-Host "  Waiting for $ServiceName (port $Port)..." -ForegroundColor Yellow -NoNewline
    
    $elapsed = 0
    $checkInterval = 1
    
    while ($elapsed -lt $TimeoutSeconds) {
        if (Test-ServicePort -Port $Port) {
                Write-Host " Ready! [OK]" -ForegroundColor Green
            return $true
        }
        
        Start-Sleep -Seconds $checkInterval
        $elapsed += $checkInterval
        Write-Host "." -NoNewline -ForegroundColor Gray
    }
    
        Write-Host " TIMEOUT! [FAILED]" -ForegroundColor Red
    return $false
}

# Clean up existing services
Write-Host "Checking for existing services..." -ForegroundColor Yellow
$portsToClean = @(8000, 8001, 8002, 8003)
$cleaned = $false

foreach ($port in $portsToClean) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        Write-Host "  Cleaning up port $port..." -ForegroundColor Yellow
        $connections | ForEach-Object {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
        $cleaned = $true
    }
}

if ($cleaned) {
    Write-Host "  Waiting for ports to be released..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

Write-Host ""

# =============================================================================
# START SERVICES IN DEPENDENCY ORDER
# =============================================================================

Write-Host "Starting services in dependency order..." -ForegroundColor Cyan
Write-Host ""

# 1. MODEL SERVICE (No dependencies - takes longest to start)
Write-Host "[1/4] Starting Model Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$projectPath'; `
    Write-Host '========================================' -ForegroundColor Yellow; `
    Write-Host '  MODEL SERVICE (Port 8002)            ' -ForegroundColor Yellow; `
    Write-Host '========================================' -ForegroundColor Yellow; `
    Write-Host ''; `
    python -m uvicorn services.model.main:app --host 0.0.0.0 --port 8002 --reload --workers 1"

if (-not (Wait-ForServiceReady -Port 8002 -ServiceName "Model" -TimeoutSeconds 30)) {
    Write-Host ""
    Write-Host "ERROR: Model service failed to start!" -ForegroundColor Red
    Write-Host "Check the Model service window for errors." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# 2. PREPROCESSING SERVICE (Depends on Model)
Write-Host "[2/4] Starting Preprocessing Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$projectPath'; `
    Write-Host '========================================' -ForegroundColor Cyan; `
    Write-Host '  PREPROCESSING SERVICE (Port 8001)    ' -ForegroundColor Cyan; `
    Write-Host '========================================' -ForegroundColor Cyan; `
    Write-Host ''; `
    python -m uvicorn services.preprocessing.main:app --host 0.0.0.0 --port 8001 --reload"

if (-not (Wait-ForServiceReady -Port 8001 -ServiceName "Preprocessing" -TimeoutSeconds 20)) {
    Write-Host ""
    Write-Host "ERROR: Preprocessing service failed to start!" -ForegroundColor Red
    Write-Host "Check the Preprocessing service window for errors." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# 3. RECEIVER SERVICE (Depends on Preprocessing)
Write-Host "[3/4] Starting Receiver Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$projectPath'; `
    Write-Host '========================================' -ForegroundColor Green; `
    Write-Host '  RECEIVER SERVICE (Port 8000)         ' -ForegroundColor Green; `
    Write-Host '========================================' -ForegroundColor Green; `
    Write-Host ''; `
    python -m uvicorn services.ble_receiver.Wifi_main_csv:app --host 0.0.0.0 --port 8000 --reload"

if (-not (Wait-ForServiceReady -Port 8000 -ServiceName "Receiver" -TimeoutSeconds 20)) {
    Write-Host ""
    Write-Host "ERROR: Receiver service failed to start!" -ForegroundColor Red
    Write-Host "Check the Receiver service window for errors." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# 4. UI SERVICE (Depends on Receiver & Preprocessing)
Write-Host "[4/4] Starting UI Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$projectPath'; `
    Write-Host '========================================' -ForegroundColor Magenta; `
    Write-Host '  UI SERVICE (Port 8003) - Gradio      ' -ForegroundColor Magenta; `
    Write-Host '========================================' -ForegroundColor Magenta; `
    Write-Host ''; `
    python -m uvicorn services.ui.main:app --host 0.0.0.0 --port 8003 --reload"

if (-not (Wait-ForServiceReady -Port 8003 -ServiceName "UI" -TimeoutSeconds 20)) {
    Write-Host ""
    Write-Host "ERROR: UI service failed to start!" -ForegroundColor Red
    Write-Host "Check the UI service window for errors." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# =============================================================================
# ALL SERVICES STARTED SUCCESSFULLY
# =============================================================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "  [OK] ALL SERVICES STARTED!           " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Service Status:" -ForegroundColor Cyan
Write-Host "  [OK] Model Service:         http://localhost:8002" -ForegroundColor White
Write-Host "  [OK] Preprocessing Service: http://localhost:8001" -ForegroundColor White
Write-Host "  [OK] Receiver Service:      http://localhost:8000" -ForegroundColor White
Write-Host "  [OK] UI Service (Gradio):   http://localhost:8003" -ForegroundColor Green
Write-Host ""

# Open UI in browser
Write-Host "Opening Gradio UI in browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process "http://localhost:8003"
Write-Host ""

# Start ngrok if requested
if (-not $SkipNgrok) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Mobile Access Setup                  " -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if ngrok is available
    $ngrokPath = $null
    if (Test-Path "C:\ngrok\ngrok.exe") {
        $ngrokPath = "C:\ngrok\ngrok.exe"
    } elseif (Get-Command ngrok -ErrorAction SilentlyContinue) {
        $ngrokPath = "ngrok"
    }
    
    if ($ngrokPath) {
        Write-Host "Starting ngrok tunnel..." -ForegroundColor Yellow
        Write-Host ""
        
        Start-Process powershell -ArgumentList `
            "-NoExit", `
            "-Command", `
            "Write-Host '========================================' -ForegroundColor Cyan; `
            Write-Host '  NGROK TUNNEL - Mobile Access         ' -ForegroundColor Cyan; `
            Write-Host '========================================' -ForegroundColor Cyan; `
            Write-Host ''; `
            Write-Host 'Copy the HTTPS URL below and open on your phone:' -ForegroundColor Yellow; `
            Write-Host ''; `
            & '$ngrokPath' http 8003"
        
            Write-Host "  [OK] Ngrok tunnel starting in new window" -ForegroundColor Green
        Write-Host "  Copy the https://...ngrok-free.dev URL for mobile access" -ForegroundColor Yellow
    } else {
            Write-Host "  [!] Ngrok not found - skipping mobile access setup" -ForegroundColor Yellow
        Write-Host "  Download from: https://ngrok.com/download" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor White
Write-Host "  System Ready!                        " -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Check Gradio UI opened in browser" -ForegroundColor White
Write-Host "  2. Connect ESP32 WiFi (ESP32-PPG-Glucose)" -ForegroundColor White
Write-Host "  3. Click 'Start Collection' in UI" -ForegroundColor White
Write-Host "  4. Place finger on sensor for 90 seconds" -ForegroundColor White
Write-Host ""
Write-Host "To Stop Services:" -ForegroundColor Yellow
Write-Host "  Close all PowerShell windows or run: .\stop.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Startup Complete! Ready to collect data." -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to close this launcher" -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
