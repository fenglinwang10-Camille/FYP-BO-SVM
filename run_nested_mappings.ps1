param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not $PythonPath) {
    if (Test-Path -LiteralPath 'D:\Anaconda3\envs\eegbo\python.exe') {
        $PythonPath = 'D:\Anaconda3\envs\eegbo\python.exe'
    } else { $PythonPath = (Get-Command python -ErrorAction Stop).Source }
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$runDirectory = Join-Path $projectRoot 'results\11_nested_mapping_comparison\split_42_inner_1729\search_42'
New-Item -Path $runDirectory -ItemType Directory -Force | Out-Null
$processFile = Join-Path $runDirectory 'background_process.json'
if (Test-Path -LiteralPath $processFile) {
    $previous = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $previous.started_utc) {
        throw "Mapping comparison is already running (PID $($previous.pid))."
    }
}
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $runDirectory "run_${timestamp}.log"
$stderrPath = Join-Path $runDirectory "run_${timestamp}.stderr.log"
$scriptPath = Join-Path $projectRoot 'nested_mapping_comparison.py'
$arguments = @('-B', '-u', ('"' + $scriptPath + '"'))
$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid=$process.Id; started_utc=$process.StartTime.ToUniversalTime().ToString('o');
   stdout=$stdoutPath; stderr=$stderrPath; python=$PythonPath } |
    ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
Write-Output "Started nested mapping comparison. PID: $($process.Id)"
Write-Output "Results: $runDirectory"
Write-Output "Log: $stdoutPath"
Write-Output 'Run this script again to resume. Keep the computer powered and awake.'
