# Mechanical file checks. An explicit -SyncTestHost enables disposable OpenSSH/SFTP fixtures.
param([string]$SyncTestHost)
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
    $rejected = $false
    try { & $tool -Action Sync -Root $temp -Manifest $manifest | Out-Null }
    catch { $rejected = $_.Exception.Message -like '*explicit -Direction*' }
    if (-not $rejected) { throw 'Sync did not require an explicit direction' }
    Write-Output 'PASS: manifest, tampering, paths, review fields, explicit direction'

    if ($SyncTestHost) {
        if ($SyncTestHost -notmatch '^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$') { throw 'Invalid test host' }
        $remote = [string](& ssh $SyncTestHost 'mktemp -d /tmp/research-sync-test.XXXXXXXX')
        if ($LASTEXITCODE -ne 0 -or $remote -cnotmatch '^/tmp/research-sync-test\.[A-Za-z0-9]{8}$') {
            throw 'Cannot identify isolated remote test directory'
        }
        $sync = @{Action='Sync';Root=$temp;Manifest=$manifest;HostName=$SyncTestHost;RemoteRoot=$remote;Direction='Push'}
        function Save-TestManifest([string[]]$names,[string]$file) {
            (& $tool -Action Manifest -Root $temp -Files $names) | Set-Content -LiteralPath $file -Encoding UTF8
        }
        try {
            $previous = Join-Path $temp 'previous.json'
            [IO.File]::WriteAllText((Join-Path $temp 'guarded.txt'), 'first')
            [IO.File]::WriteAllText((Join-Path $temp 'same.txt'), 'same')
            Save-TestManifest @('guarded.txt') $previous
            Save-TestManifest @('./guarded.txt','same.txt') $manifest
            $initial = & $tool @sync | ConvertFrom-Json
            if (($initial.actions -join ',') -ne 'create,create') { throw 'Initial publication failed' }

            [IO.File]::WriteAllText((Join-Path $temp 'guarded.txt'), 'other')
            [IO.File]::WriteAllText((Join-Path $temp 'fresh.txt'), 'fresh')
            Save-TestManifest @('fresh.txt','guarded.txt','same.txt') $manifest
            $rejected = $false
            try { & $tool @sync | Out-Null } catch { $rejected = $_.Exception.Message -like '*prepare failed*' }
            if (-not $rejected) { throw 'Existing conflicting target accepted without previous identity' }

            $stale = Join-Path $temp 'stale.json'
            Save-TestManifest @('guarded.txt') $stale
            $rejected = $false
            try { & $tool @sync -ExpectedRemoteManifest $stale | Out-Null }
            catch { $rejected = $_.Exception.Message -like '*prepare failed*' }
            if (-not $rejected) { throw 'Wrong old SHA accepted for same-size target' }

            $written = & $tool @sync -ExpectedRemoteManifest $previous | ConvertFrom-Json
            if (($written.actions -join ',') -ne 'create,replace,skip' -or -not $written.all_size_sha256_match) {
                throw 'A refused batch changed a file, or authorized replacement/readback failed'
            }
            $repeat = & $tool @sync -ExpectedRemoteManifest $previous | ConvertFrom-Json
            if (($repeat.actions -join ',') -ne 'skip,skip,skip') { throw 'Verified retry did not skip existing files' }
            $pullRoot = (New-Item -ItemType Directory -Path (Join-Path $temp 'pulled')).FullName
            $pulled = & $tool -Action Sync -Direction Pull -Root $pullRoot -Manifest $manifest -HostName $SyncTestHost -RemoteRoot $remote | ConvertFrom-Json
            $verified = & $tool -Action Verify -Root $pullRoot -Manifest $manifest | ConvertFrom-Json
            if (-not $pulled.all_size_sha256_match -or -not $verified.all_size_sha256_match) { throw 'Pull did not install verified files' }
            Write-Output 'PASS: create, whole-batch refusal, stale SHA, authorized replacement, identical skip, retry, pull/readback'
        }
        finally {
            $batch = Join-Path $temp 'cleanup.batch'
            [IO.File]::WriteAllLines($batch, @("-rm $remote/guarded.txt","-rm $remote/same.txt","-rm $remote/fresh.txt","rmdir $remote"), [Text.UTF8Encoding]::new($false))
            & sftp -oBatchMode=no -b $batch $SyncTestHost | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Remote fixture cleanup incomplete: $remote" }
        }
    }
}
finally {
    if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($temp)) -ne [IO.Path]::GetTempPath().TrimEnd([IO.Path]::DirectorySeparatorChar)) {
        throw 'Unsafe cleanup target'
    }
    Remove-Item -LiteralPath $temp -Recurse -Force
}
