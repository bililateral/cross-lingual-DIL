# Mechanical file checks only; no project Python, research execution or network.
$ErrorActionPreference = 'Stop'
$tool = Join-Path $PSScriptRoot '../scripts/research_files.ps1'
$temp = Join-Path ([IO.Path]::GetTempPath()) ('research-files-test-' + [guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $temp)
try {
    [IO.File]::WriteAllText((Join-Path $temp 'sample.md'), 'fixed evidence')
    $manifest = Join-Path $temp 'manifest.json'
    (& $tool -Action Manifest -Root $temp -Files sample.md) | Set-Content -LiteralPath $manifest -Encoding utf8
    $saved = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    if ($saved.bytes -ne [Text.Encoding]::UTF8.GetByteCount('fixed evidence')) { throw 'Manifest total is not the actual byte sum' }
    $pass = & $tool -Action Verify -Root $temp -Manifest $manifest | ConvertFrom-Json
    if (-not $pass.all_size_sha256_match) { throw 'Valid identity rejected' }
    [IO.File]::WriteAllText((Join-Path $temp 'sample.md'), 'tampered')
    $rejected = $false
    try { & $tool -Action Verify -Root $temp -Manifest $manifest | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Tampered bytes accepted' }
    $rejected = $false
    try { & $tool -Action Manifest -Root $temp -Files '../outside.md' | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Traversal accepted' }
    $rejected = $false
    try { & $tool -Action ReviewRecord -Root $temp -Files sample.md | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Review record accepted without primary disposition' }
    Write-Output 'PASS: identity roundtrip, tamper rejection, path rejection, review-field rejection'
}
finally {
    if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($temp)) -ne [IO.Path]::GetTempPath().TrimEnd([IO.Path]::DirectorySeparatorChar)) {
        throw 'Unsafe cleanup target'
    }
    Remove-Item -LiteralPath $temp -Recurse -Force
}
