param(
    [int]$OuterSplitSeed = 42,
    [int]$InnerSplitSeed = 1729,
    [int]$SearchSeed = 42,
    [string]$PythonPath = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not $PythonPath) {
    $preferredPython = 'D:\Anaconda3\envs\eegbo\python.exe'
    if (Test-Path -LiteralPath $preferredPython) {
        $PythonPath = $preferredPython
    } else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$outputRoot = Join-Path $projectRoot 'results\10_nested_cv_shared_initialization'
$runDirectory = Join-Path $outputRoot "split_${OuterSplitSeed}_inner_${InnerSplitSeed}\search_${SearchSeed}"
New-Item -Path $runDirectory -ItemType Directory -Force | Out-Null
$processFile = Join-Path $runDirectory 'background_process.json'
if (Test-Path -LiteralPath $processFile) {
    $previous = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $previous.started_utc) {
        throw "This experiment is already running (PID $($previous.pid))."
    }
}

# Limit BLAS threads: the experiment runs one fold at a time to control memory.
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $runDirectory "run_${timestamp}.log"
$stderrPath = Join-Path $runDirectory "run_${timestamp}.stderr.log"
$scriptPath = Join-Path $projectRoot 'nested_bo_rs.py'
$arguments = @('-B', '-u', ('"' + $scriptPath + '"'),
    '--outer-split-seed', $OuterSplitSeed, '--inner-split-seed', $InnerSplitSeed,
    '--search-seed', $SearchSeed, '--output-dir', ('"' + $outputRoot + '"'))
$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{
    pid = $process.Id
    started_utc = $process.StartTime.ToUniversalTime().ToString('o')
    stdout = $stdoutPath
    stderr = $stderrPath
    python = $PythonPath
} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
Write-Output "Started nested CV in the background. PID: $($process.Id)"
Write-Output "Results: $runDirectory"
Write-Output "Progress log: $stdoutPath"
Write-Output 'Run this script again to resume after interruption. Keep the computer awake.'
