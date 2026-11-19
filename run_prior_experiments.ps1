$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\Anaconda3\envs\eegbo\python.exe"
$comparisonDir = Join-Path $projectDir "results\03_prior_comparison_nf_enhanced"
$logPath = Join-Path $comparisonDir "prior_experiments.log"

Set-Location $projectDir
New-Item -ItemType Directory -Force (Split-Path -Parent $logPath) | Out-Null

foreach ($prior in @("uniform", "normal", "exponential", "gamma")) {
    $outputDir = Join-Path $comparisonDir $prior
    $historyPath = Join-Path $outputDir "bo_history.csv"
    $bestPath = Join-Path $outputDir "bo_best_params.csv"
    if ((Test-Path $historyPath) -and (Test-Path $bestPath)) {
        $rows = @(Import-Csv $historyPath)
        $validRows = @($rows | Where-Object {
            $_.feature_setting -eq "nf_enhanced" -and $_.parameter_prior -eq $prior
        })
        if ($rows.Count -eq 80 -and $validRows.Count -eq 80) {
            "[{0}] Skipping completed {1} prior" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $prior |
                Tee-Object -FilePath $logPath -Append
            continue
        }
        throw "Existing completed files for $prior do not match the comparison setting: $outputDir"
    }
    $startMessage = "[{0}] Starting {1} prior" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $prior
    $startMessage | Tee-Object -FilePath $logPath -Append

    & $python -u "bo_optimizer.py" --parameter-prior $prior --feature-setting nf_enhanced --output-dir $outputDir *>&1 |
        Tee-Object -FilePath $logPath -Append

    if ($LASTEXITCODE -ne 0) {
        throw "$prior prior experiment failed with exit code $LASTEXITCODE"
    }

    $endMessage = "[{0}] Completed {1} prior" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $prior
    $endMessage | Tee-Object -FilePath $logPath -Append
}
