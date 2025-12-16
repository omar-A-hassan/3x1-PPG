# Optimized PPG Service Launcher
# Starts all services in correct dependency order with health checks
# Version 3.0 - With Authentication Service

param(
    [switch]$SkipNgrok = $true
)

$ErrorActionPreference = "Continue"
# Use the directory containing this script as the project root
$projectPath = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PPG Monitoring System Launcher v3.0  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

Write-Host "Pre-flight checks..." -ForegroundColor Yellow

# Check if MySQL is running (optional - skip auth service if not available)
Write-Host "  Checking MySQL service..." -ForegroundColor Yellow -NoNewline
$mysqlService = Get-Service -Name "MySQL*" -ErrorAction SilentlyContinue
$skipAuth = $false
if ($mysqlService -and $mysqlService.Status -eq "Running") {
    Write-Host " Running [OK]" -ForegroundColor Green
} else {
    Write-Host " NOT FOUND" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  MySQL not installed or not running." -ForegroundColor Yellow
    Write-Host "  Auth service will be SKIPPED - using mock mode in UI." -ForegroundColor Yellow
    Write-Host ""
    $skipAuth = $true
}

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
$portsToClean = @(8000, 8001, 8002, 8003, 8004)
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

# 1. AUTH SERVICE (Only if MySQL is available)
if (-not $skipAuth) {
    Write-Host "[1/5] Starting Auth Service..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList `
        "-NoExit", `
        "-Command", `
        "cd '$projectPath'; `
        Write-Host '========================================' -ForegroundColor Blue; `
        Write-Host '  AUTH SERVICE (Port 8004)             ' -ForegroundColor Blue; `
        Write-Host '========================================' -ForegroundColor Blue; `
        Write-Host ''; `
        python -m uvicorn services.auth_service.auth_service:app --host 0.0.0.0 --port 8004 --reload"

    if (-not (Wait-ForServiceReady -Port 8004 -ServiceName "Auth" -TimeoutSeconds 20)) {
        Write-Host ""
        Write-Host "WARNING: Auth service failed to start!" -ForegroundColor Yellow
        Write-Host "Continuing without auth - UI will use mock mode." -ForegroundColor Yellow
        $skipAuth = $true
    }
    Write-Host ""
} else {
    Write-Host "[1/5] Skipping Auth Service (MySQL not available)..." -ForegroundColor Yellow
    Write-Host "  UI will use MOCK_AUTH_SERVICE mode" -ForegroundColor Gray
    Write-Host ""
}

# 2. PREPROCESSING SERVICE (No dependencies)
Write-Host "[2/5] Starting Preprocessing Service..." -ForegroundColor Yellow
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

# 3. MODEL SERVICE (No dependencies - loads model at startup)
Write-Host "[3/5] Starting Model Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$projectPath'; `
    Write-Host '========================================' -ForegroundColor DarkYellow; `
    Write-Host '  MODEL SERVICE (Port 8002)            ' -ForegroundColor DarkYellow; `
    Write-Host '========================================' -ForegroundColor DarkYellow; `
    Write-Host ''; `
    python -m uvicorn services.model.main:app --host 0.0.0.0 --port 8002 --reload"

if (-not (Wait-ForServiceReady -Port 8002 -ServiceName "Model" -TimeoutSeconds 30)) {
    Write-Host ""
    Write-Host "ERROR: Model service failed to start!" -ForegroundColor Red
    Write-Host "Check the Model service window for errors." -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

# 4. RECEIVER SERVICE (Depends on Preprocessing & Model)
Write-Host "[4/5] Starting Receiver Service..." -ForegroundColor Yellow
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

# 5. UI SERVICE (Depends on Receiver, Preprocessing, Model & Auth)
Write-Host "[5/5] Starting UI Service..." -ForegroundColor Yellow
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
if (-not $skipAuth) {
    Write-Host "  [OK] Auth Service:          http://localhost:8004" -ForegroundColor White
} else {
    Write-Host "  [--] Auth Service:          SKIPPED (using mock mode)" -ForegroundColor Gray
}
Write-Host "  [OK] Preprocessing Service: http://localhost:8001" -ForegroundColor White
Write-Host "  [OK] Model Service:         http://localhost:8002" -ForegroundColor White
Write-Host "  [OK] Receiver Service:      http://localhost:8000" -ForegroundColor White
Write-Host "  [OK] UI Service (Gradio):   http://localhost:8003" -ForegroundColor Green
Write-Host ""

# Open UI in browser
Write-Host "Opening Gradio UI in browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process "http://localhost:8003"
Write-Host ""

# Start Cloudflare tunnels if requested
if (-not $SkipNgrok) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Remote Access Setup (Cloudflare)     " -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if cloudflared is available
    $cloudflaredPath = $null
    if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
        $cloudflaredPath = "cloudflared"
    }
    
    if ($cloudflaredPath) {
        Write-Host "Starting Cloudflare tunnels..." -ForegroundColor Yellow
        Write-Host ""
        
        # Start UI Tunnel (port 8003)
        Write-Host "  Starting UI tunnel (port 8003)..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList `
            "-NoExit", `
            "-Command", `
            "Write-Host '========================================' -ForegroundColor Cyan; `
            Write-Host '  CLOUDFLARE TUNNEL - UI (Port 8003)   ' -ForegroundColor Cyan; `
            Write-Host '========================================' -ForegroundColor Cyan; `
            Write-Host ''; `
            Write-Host 'Copy the URL below for mobile/remote access:' -ForegroundColor Yellow; `
            Write-Host ''; `
            cloudflared tunnel --url http://localhost:8003"
        
        Write-Host "  Waiting 12 seconds before starting second tunnel..." -ForegroundColor Gray
        Start-Sleep -Seconds 12
        
        # Start Receiver Tunnel (port 8000 - for ESP32)
        Start-Process powershell -ArgumentList `
            "-NoExit", `
            "-Command", `
            "Write-Host '========================================' -ForegroundColor Green; `
            Write-Host '  CLOUDFLARE TUNNEL - ESP32 (Port 8000)' -ForegroundColor Green; `
            Write-Host '========================================' -ForegroundColor Green; `
            Write-Host ''; `
            Write-Host 'Copy the URL below and configure in ESP32:' -ForegroundColor Yellow; `
            Write-Host '(Format: https://your-url.trycloudflare.com)' -ForegroundColor Gray; `
            Write-Host ''; `
            cloudflared tunnel --url http://localhost:8000"
        
        Write-Host "  [OK] UI tunnel starting in new window" -ForegroundColor Green
        Write-Host "  [OK] Receiver tunnel starting in new window" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Copy the URLs from each tunnel window:" -ForegroundColor Yellow
        Write-Host "    - UI tunnel: Use for mobile/remote access" -ForegroundColor White
        Write-Host "    - Receiver tunnel: Configure in ESP32 settings" -ForegroundColor White
    } else {
        Write-Host "  [!] Cloudflared not found - skipping remote access setup" -ForegroundColor Yellow
        Write-Host "  Install with: winget install cloudflare.cloudflared" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor White
Write-Host "  System Ready!                        " -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Check Gradio UI opened in browser" -ForegroundColor White
Write-Host "  2. Login or Register in the Login tab" -ForegroundColor White
Write-Host "  3. Copy your API key for ESP32" -ForegroundColor White
if (-not $SkipNgrok) {
    Write-Host "  4. Copy tunnel URLs from the tunnel windows:" -ForegroundColor White
    Write-Host "     - UI tunnel: For mobile/remote access" -ForegroundColor Gray
    Write-Host "     - Receiver tunnel: Configure in ESP32" -ForegroundColor Gray
    Write-Host "  5. Configure ESP32 with receiver tunnel URL + API key" -ForegroundColor White
    Write-Host "  6. Connect ESP32 and place finger on sensor" -ForegroundColor White
} else {
    Write-Host "  4. Connect ESP32 and place finger on sensor" -ForegroundColor White
}
Write-Host ""
Write-Host "To Stop Services:" -ForegroundColor Yellow
Write-Host "  Close all PowerShell windows or run: .\stop.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Startup Complete! Ready to collect data." -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to close this launcher" -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
