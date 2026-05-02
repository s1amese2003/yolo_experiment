param(
    [string]$Device = "",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Device -eq "") {
    if ($Quick) {
        python .\src\train_experiments.py --quick
    } else {
        python .\src\train_experiments.py
    }
} else {
    if ($Quick) {
        python .\src\train_experiments.py --device $Device --quick
    } else {
        python .\src\train_experiments.py --device $Device
    }
}
