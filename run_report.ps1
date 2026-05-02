$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\src\benchmark_speed.py --split val --samples 100
python .\src\summarize_results.py
