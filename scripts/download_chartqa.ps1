# Download ChartQA Dataset from HuggingFace and extract only JSON files.
# Images are ~850 MB and will be downloaded separately on Ada for inference.
# The JSON files are all we need for split statistics and manifest work.

param(
    [string]$OutDir = "data\ChartQA"
)

$ErrorActionPreference = "Stop"

$zipUrl = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
$zipFile = Join-Path $OutDir "ChartQA_Dataset.zip"

# Create output directory
New-Item -ItemType Directory -Force $OutDir | Out-Null
Write-Host "Downloading ChartQA Dataset (~875 MB)..."
Write-Host "URL: $zipUrl"

# Download
Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile -UseBasicParsing

Write-Host "Download complete. Extracting JSON files only..."

# Extract - PowerShell can selectively extract from zip
Add-Type -Assembly System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipFile)

$jsonCount = 0
foreach ($entry in $zip.Entries) {
    if ($entry.Name -match '\.json$' -and $entry.Name -notmatch 'annotation') {
        # Preserve directory structure relative to the zip root
        $relPath = $entry.FullName
        # ChartQA zip typically has "ChartQA Dataset/" prefix
        $relPath = $relPath -replace '^ChartQA Dataset/', ''
        $destPath = Join-Path $OutDir $relPath
        $destDir = Split-Path $destPath -Parent
        
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Force $destDir | Out-Null
        }
        
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
        Write-Host "  Extracted: $relPath"
        $jsonCount++
    }
}

$zip.Dispose()

Write-Host "`nExtracted $jsonCount JSON files to $OutDir"
Write-Host "Removing zip file to save space..."
Remove-Item $zipFile
Write-Host "Done."
