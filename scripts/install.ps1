param(
  [string]$Version = "",
  [string]$Repo = "ModelsLab/fusion",
  [string]$Destination = "$env:LocalAppData\Programs\fusion"
)

$ErrorActionPreference = "Stop"

function Get-Architecture {
  $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
  switch ($arch) {
    "x64" { return "amd64" }
    "arm64" { return "arm64" }
    default { throw "unsupported architecture: $arch" }
  }
}

function Resolve-Version {
  param([string]$RepoName, [string]$RequestedVersion)

  if ($RequestedVersion) {
    return $RequestedVersion
  }

  $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$RepoName/releases/latest"
  if (-not $release.tag_name) {
    throw "failed to resolve the latest release for $RepoName"
  }

  return $release.tag_name
}

$arch = Get-Architecture
$resolvedVersion = Resolve-Version -RepoName $Repo -RequestedVersion $Version
$versionNoV = $resolvedVersion.TrimStart("v")
$asset = "fusion_${versionNoV}_windows_${arch}.zip"
$baseUrl = "https://github.com/$Repo/releases/download/$resolvedVersion"
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("fusion-install-" + [guid]::NewGuid().ToString("N"))

New-Item -ItemType Directory -Path $tmpDir | Out-Null

try {
  $assetPath = Join-Path $tmpDir $asset
  $checksumsPath = Join-Path $tmpDir "checksums.txt"

  Invoke-WebRequest -Uri "$baseUrl/$asset" -OutFile $assetPath
  Invoke-WebRequest -Uri "$baseUrl/checksums.txt" -OutFile $checksumsPath

  $expected = (Get-Content $checksumsPath | Where-Object { $_ -match [regex]::Escape($asset) + '$' } | Select-Object -First 1).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)[0].ToLowerInvariant()
  $actual = (Get-FileHash -Path $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($expected -ne $actual) {
    throw "checksum mismatch for $asset"
  }

  Expand-Archive -Path $assetPath -DestinationPath $tmpDir -Force
  New-Item -ItemType Directory -Path $Destination -Force | Out-Null
  Copy-Item -Path (Join-Path $tmpDir "fusion.exe") -Destination (Join-Path $Destination "fusion.exe") -Force

  Write-Host "installed fusion to $(Join-Path $Destination 'fusion.exe')"
  if (-not (($env:PATH -split ';') -contains $Destination)) {
    Write-Host "add $Destination to PATH if it is not already exported"
  }
}
finally {
  Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
