# Extract the CT archives into one data folder per season. Handles both layouts seen so far:
#   2526: one archive per box, set-<id>-2526_<box>.tar
#   2425: one archive for the whole season, set-<id>-2425.tar
#
# Usage, from any folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\extract_scans.ps1                # season 2526
#   powershell -ExecutionPolicy Bypass -File scripts\extract_scans.ps1 -Season 2425
#
# Result: all .nii files of one season in <Target>\<Season>\ (no subfolders).
# The .tar files in Downloads are not changed or deleted. Running it twice is safe.
param(
    [string]$Season = "2526",
    [string]$Source = "$env:USERPROFILE\Downloads",
    [string]$Target = "$env:USERPROFILE\thesis-data\ct"
)
$ErrorActionPreference = "Stop"

$archives = Get-ChildItem -Path $Source -Filter "set-*-${Season}*.tar" | Sort-Object Name
if (-not $archives) {
    throw "No archives named set-*-${Season}*.tar found in $Source"
}

$dest = Join-Path $Target $Season
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Write-Host "Extracting $($archives.Count) archives to $dest"

$expected = 0
foreach ($archive in $archives) {
    # Count the volumes; strip the top folder only if every volume sits inside one
    $inArchive = @(tar -tf $archive.FullName | Where-Object { $_ -like "*.nii" -or $_ -like "*.nii.gz" })
    $expected += $inArchive.Count
    $nested = @($inArchive | Where-Object { $_ -like "*/*" }).Count -eq $inArchive.Count
    if ($nested) {
        tar -xf $archive.FullName -C $dest --strip-components=1
    } else {
        tar -xf $archive.FullName -C $dest
    }
    if ($LASTEXITCODE -ne 0) { throw "tar failed on $($archive.Name)" }
    Write-Host ("  {0}: {1} volumes" -f $archive.Name, $inArchive.Count)
}

# Check: every volume arrived, and every file has the size of a 128^3 16-bit NIfTI volume
$files = @(Get-ChildItem -Path $dest -File | Where-Object { $_.Name -like "*.nii" -or $_.Name -like "*.nii.gz" })
Write-Host ""
Write-Host "Volumes in archives: $expected"
Write-Host "Volumes in ${dest}: $($files.Count)"
if ($files.Count -lt $expected) {
    Write-Warning "Fewer files than expected. Two archives may contain the same file name."
}
$odd = @($files | Where-Object { $_.Name -like "*.nii" -and $_.Length -ne 4194656 })
if ($odd.Count -gt 0) {
    Write-Warning "$($odd.Count) files do not have the expected size of 4,194,656 bytes:"
    $odd | Select-Object Name, Length | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Done. Put this line in config\local.toml, under [paths]:"
Write-Host ("ct_dir_{0} = ""{1}""" -f $Season, ($dest -replace '\\', '/'))
