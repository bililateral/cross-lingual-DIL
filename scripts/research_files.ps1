# Mechanical file custody only. Credentials are handled by OpenSSH, never this script.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Manifest','Verify','Sync','ReviewRecord')][string]$Action,
    [Parameter(Mandatory)][string]$Root,
    [string[]]$Files,
    [string]$Manifest,
    [string]$HostName,
    [string]$RemoteRoot,
    [ValidateSet('Push','Pull')][string]$Direction = 'Push',
    [string]$ReviewUrl,
    [string]$Disposition,
    [string]$BrowserOutcome
)
$ErrorActionPreference = 'Stop'
$rootPath = (Resolve-Path -LiteralPath $Root).Path
function Get-Target([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative -match '[\r\n"\\]' -or
        $Relative.Split('/') -contains '..') { throw "Use a relative forward-slash path: $Relative" }
    $full = [IO.Path]::GetFullPath((Join-Path $rootPath $Relative))
    if (-not $full.StartsWith($rootPath.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar,
                            [StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside root' }
    return $full
}
function Get-Identity([string]$Relative) {
    $file = Get-Item -LiteralPath (Get-Target $Relative)
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Regular file required' }
    return [ordered]@{ path=$Relative; bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower() }
}
function Read-Manifest {
    if (-not $Manifest) { throw '-Manifest is required' }
    $data = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    if ($data -isnot [pscustomobject] -or -not $data.files -or @($data.files).Count -eq 0) { throw 'Manifest requires a nonempty files array' }
    $seen = @{}
    foreach ($row in $data.files) {
        [void](Get-Target $row.path)
        if ($row.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or $null -eq $row.bytes -or $row.bytes -lt 0 -or $seen.ContainsKey($row.path)) {
            throw 'Invalid or duplicate file identity'
        }
        $seen[$row.path] = $true
    }
    return $data
}
function Assert-File($Row, [string]$Path) {
    if ((Get-Item -LiteralPath $Path).Length -ne $Row.bytes -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Row.sha256) { throw "Size/SHA mismatch: $($Row.path)" }
}
if ($Action -in 'Manifest','ReviewRecord') {
    if (-not $Files -or @($Files | Select-Object -Unique).Count -ne $Files.Count) { throw 'Explicit, unique -Files are required' }
    $rows = @($Files | ForEach-Object { Get-Identity $_ })
    $record = [ordered]@{ recorded_at=(Get-Date -Format o); files=$rows; count=$rows.Count;
        bytes=($rows | ForEach-Object { [long]$_.bytes } | Measure-Object -Sum).Sum }
    if ($Action -eq 'ReviewRecord') {
        if (-not $ReviewUrl -or -not $Disposition -or -not $BrowserOutcome) { throw 'Supply review URL, primary disposition path and observed browser outcome' }
        [void](Get-Identity $Disposition)
        $record.review_url = $ReviewUrl
        $record.primary_disposition = Get-Identity $Disposition
        $record.browser_outcome = $BrowserOutcome
        $record.scope = 'File identities only; does not certify review acceptance, cleanup or scientific success'
    }
    $record | ConvertTo-Json -Depth 8
    exit 0
}
$data = Read-Manifest
if ($Action -eq 'Verify') {
    foreach ($row in $data.files) { Assert-File $row (Get-Target $row.path) }
    [ordered]@{ verified_at=(Get-Date -Format o); count=@($data.files).Count; all_size_sha256_match=$true } | ConvertTo-Json
    exit 0
}
if (-not $HostName -or $HostName -notmatch '^[A-Za-z0-9_.@-]+$' -or $HostName.StartsWith('-') -or
    -not $RemoteRoot -or $RemoteRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteRoot.Split('/') -contains '..') {
    throw 'Provide an SSH host and explicit absolute remote root (safe path characters only)'
}
$temp = Join-Path $rootPath ('.research-sync-' + [guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $temp)
try {
    $commands = [Collections.Generic.List[string]]::new()
    $i = 0
    foreach ($row in $data.files) {
        # Explicit review/source/metadata lists only; never bulk-transfer research payloads.
        if ($row.path -match '(?i)(private_custody|(^|/)(owners|qrels|labels)([./]|$)|\.(pt|bin|npy|npz|safetensors|sql|jsonl)$)') {
            throw "Research payload excluded: $($row.path)"
        }
        $local = (Get-Target $row.path).Replace('\','/')
        $remote = $RemoteRoot.TrimEnd('/') + '/' + $row.path
        $back = (Join-Path $temp "$i.readback").Replace('\','/')
        if ($Direction -eq 'Push') {
            Assert-File $row $local
            $commands.Add('put "' + $local + '" "' + $remote + '"')
        }
        $commands.Add('get "' + $remote + '" "' + $back + '"')
        $i++
    }
    $batch = Join-Path $temp 'sftp.batch'
    [IO.File]::WriteAllText($batch, ($commands -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
    # BatchMode=no permits OpenSSH's own terminal authentication prompt; no password parameter.
    & sftp -oBatchMode=no -b $batch $HostName | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'SFTP failed; no successful receipt is issued' }
    $i = 0
    foreach ($row in $data.files) { Assert-File $row (Join-Path $temp "$i.readback"); $i++ }
    if ($Direction -eq 'Pull') {
        # Verify the entire batch before installing any local result; preserve conflicting files.
        foreach ($row in $data.files) {
            $target = Get-Target $row.path
            if (Test-Path -LiteralPath $target) { Assert-File $row $target }
        }
        $i = 0
        foreach ($row in $data.files) {
            $target = Get-Target $row.path
            if (-not (Test-Path -LiteralPath $target)) {
                [void](New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force)
                Move-Item -LiteralPath (Join-Path $temp "$i.readback") -Destination $target
            }
            $i++
        }
    }
    [ordered]@{ verified_at=(Get-Date -Format o); transport='SFTP'; direction=$Direction;
        remote_root=$RemoteRoot; files=$data.files; count=@($data.files).Count;
        all_size_sha256_match=$true; verification='SFTP readback size/SHA';
        note='No remote directories created or deleted; push targets must be authorized mutable/new files' } | ConvertTo-Json -Depth 8
}
finally {
    # Only this invocation's unique local directory; resolved boundary checked before recursive cleanup.
    $resolved = [IO.Path]::GetFullPath($temp)
    if ([IO.Path]::GetDirectoryName($resolved) -ne $rootPath -or [IO.Path]::GetFileName($resolved) -notlike '.research-sync-*') {
        throw 'Unsafe temporary cleanup target'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
