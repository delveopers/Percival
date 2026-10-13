# clean_pycache.ps1
# Deletes all __pycache__ directories under the current project.
# run: powershell -ExecutionPolicy Bypass -File .\clean_pycache.ps1

$ErrorActionPreference = "Stop"

$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

Write-Host "Searching for __pycache__ folders under:" -ForegroundColor Cyan
Write-Host $root
Write-Host ""

$folders = Get-ChildItem -Path $root -Directory -Recurse -Force |
  Where-Object { $_.Name -eq "__pycache__" }

if (-not $folders) {
  Write-Host "No __pycache__ folders found." -ForegroundColor Green
  exit 0
}

$deleted = 0

foreach ($folder in $folders) {
  try {
    Remove-Item -LiteralPath $folder.FullName -Recurse -Force
    Write-Host "Deleted: $($folder.FullName)" -ForegroundColor Yellow
    $deleted++
  }
  catch {
    Write-Warning "Could not delete: $($folder.FullName)"
    Write-Warning $_.Exception.Message
  }
}

Write-Host ""
Write-Host "Deleted $deleted __pycache__ folder(s)." -ForegroundColor Green