param([string]$PythonPath='D:\Anaconda3\envs\eegbo\python.exe')
$ErrorActionPreference='Stop'
$runDirectory=Join-Path $PSScriptRoot 'results/14_training_search_extension_100/split_42_inner_1729/search_42'
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$processFile=Join-Path $runDirectory 'background_process.json'
if (Test-Path -LiteralPath $processFile) {
    $previous=Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $existing=Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($existing) {
        $savedStart=([datetimeoffset]$previous.started_utc).UtcDateTime
        if ([math]::Abs(($existing.StartTime.ToUniversalTime()-$savedStart).TotalSeconds) -lt 1) {
            throw "Search extension is already running (PID $($previous.pid))."
        }
    }
}
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath=Join-Path $runDirectory "run_$timestamp.log"
$stderrPath=Join-Path $runDirectory "run_$timestamp.stderr.log"
$process=Start-Process -FilePath $PythonPath -ArgumentList @('-u','-B','extend_nested_search.py','--budget','100') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru
@{pid=$process.Id;started_utc=$process.StartTime.ToUniversalTime().ToString('o');stdout=$stdoutPath;stderr=$stderrPath} |
    ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
Write-Output "Started training-only 100-evaluation continuation. PID: $($process.Id)"
Write-Output "Log: $stdoutPath"
