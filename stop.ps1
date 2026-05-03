# Stop All PPG Services
# Safely shuts down all running services

Write-Host "========================================" -ForegroundColor Red
Write-Host "  Stopping PPG Monitoring Services     " -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

$ports = @(
    @{Port=8000; Name="Receiver"},
    @{Port=8001; Name="Preprocessing"},
    @{Port=8002; Name="Model"},
    @{Port=8003; Name="UI"}
)

$stoppedAny = $false

foreach ($service in $ports) {
    Write-Host "Checking $($service.Name) (port $($service.Port))..." -ForegroundColor Yellow
    
    $connections = Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue
    
    if ($connections) {
        $processes = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        
        foreach ($pid in $processes) {
            try {
                $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Host "  Stopping $($proc.ProcessName) (PID: $pid)..." -ForegroundColor Yellow
                    Stop-Process -Id $pid -Force
                    $stoppedAny = $true
                    Write-Host "  [OK] Stopped" -ForegroundColor Green
                }
            } catch {
                Write-Host "  [FAIL] Failed to stop PID $pid" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  Not running" -ForegroundColor Gray
    }
}

Write-Host ""

if ($stoppedAny) {
    Write-Host "Waiting for cleanup..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  [OK] All Services Stopped            " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Gray
    Write-Host "  No services were running             " -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Gray
}

Write-Host ""
Write-Host "You can now restart services with: .\start.ps1" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close this window"
