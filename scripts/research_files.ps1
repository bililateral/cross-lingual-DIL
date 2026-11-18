# Mechanical file custody only. Credentials are handled by OpenSSH, never this script.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Manifest','Verify','Sync','ReviewRecord')][string]$Action,
    [Parameter(Mandatory)][string]$Root,
    [string[]]$Files,
    [string]$Manifest,
    [string]$HostName,
    [string]$RemoteRoot,
    [ValidateSet('Push','Pull')][string]$Direction,
    [string]$ExpectedRemoteManifest,
    [string]$ReviewUrl,
    [string]$Disposition,
    [string]$BrowserOutcome
)
$ErrorActionPreference = 'Stop'
if ($Action -eq 'Sync' -and -not $Direction) { throw 'Sync requires explicit -Direction Push or Pull' }
if ($ExpectedRemoteManifest -and ($Action -ne 'Sync' -or $Direction -ne 'Push')) {
    throw '-ExpectedRemoteManifest applies only to Push'
}
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$rootPrefix = $rootPath.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
function Get-Target([string]$Relative) {
    if (-not $Relative -or [IO.Path]::IsPathRooted($Relative) -or $Relative -match '[\x00-\x1f"\\:]' -or
        $Relative.Split('/') -contains '..') { throw "Use a relative forward-slash path: $Relative" }
    $full = [IO.Path]::GetFullPath((Join-Path $rootPath $Relative))
    if (-not $full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside root' }
    return $full
}
function Get-Identity([string]$Relative) {
    $file = Get-Item -LiteralPath (Get-Target $Relative)
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Regular file required' }
    return [ordered]@{ path=$Relative; bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower() }
}
function Read-Manifest([string]$Source) {
    if (-not $Source) { throw '-Manifest is required' }
    $data = Get-Content -LiteralPath $Source -Encoding UTF8 -Raw | ConvertFrom-Json
    if ($data -isnot [pscustomobject] -or $data.files -isnot [array] -or $data.files.Count -eq 0) { throw 'Manifest requires a nonempty files array' }
    $seen = @{}
    foreach ($row in $data.files) {
        $full = Get-Target $row.path
        if ($row.path -isnot [string] -or $row.sha256 -isnot [string] -or $row.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
            ($row.bytes -isnot [int] -and $row.bytes -isnot [long]) -or $row.bytes -lt 0 -or $seen.ContainsKey($full)) {
            throw 'Invalid or duplicate file identity'
        }
        $seen[$full] = $true
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
$data = Read-Manifest $Manifest
if ($Action -eq 'Verify') {
    foreach ($row in $data.files) { Assert-File $row (Get-Target $row.path) }
    [ordered]@{ verified_at=(Get-Date -Format o); count=@($data.files).Count; all_size_sha256_match=$true } | ConvertTo-Json
    exit 0
}
if (-not $HostName -or $HostName -notmatch '^[A-Za-z0-9_.@-]+$' -or $HostName.StartsWith('-') -or
    -not $RemoteRoot -or $RemoteRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteRoot.Split('/') -contains '..') {
    throw 'Provide an SSH host and explicit absolute remote root (safe path characters only)'
}
$token = [guid]::NewGuid().ToString('N')
$temp = Join-Path $rootPath ('.research-sync-' + $token)
function Invoke-SftpBatch([string[]]$Commands) {
    $batch = Join-Path $temp 'sftp.batch'
    [IO.File]::WriteAllLines($batch, $Commands, [Text.UTF8Encoding]::new($false))
    & sftp -oBatchMode=no -b $batch $HostName | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'SFTP failed; no successful receipt is issued' }
}
function Invoke-PushStep([string]$Phase, [int[]]$Pending) {
    $request = @{ phase=$Phase; token=$token; root=$RemoteRoot; files=$pushRows; pending=@($Pending) }
    $json = $request | ConvertTo-Json -Depth 6 -Compress
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
    $program = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'research_sync_push.py'))
    $program += "`nimport base64`nprint(json.dumps(push(json.loads(base64.b64decode('$encoded')))))"
    $reply = $program | & ssh -oBatchMode=no $HostName /home/yongpeng/miniconda3/envs/py310/bin/python -B -
    if ($LASTEXITCODE -ne 0) { throw "Remote Push $Phase failed" }
    return ($reply | ConvertFrom-Json)
}
[void](New-Item -ItemType Directory -Path $temp)
try {
    $previous = @{}
    if ($ExpectedRemoteManifest) {
        foreach ($row in (Read-Manifest $ExpectedRemoteManifest).files) {
            if ($data.files.path -cnotcontains $row.path) { throw "Previous identity outside this Push: $($row.path)" }
            $previous[$row.path] = @{ bytes=$row.bytes; sha256=$row.sha256 }
        }
    }
    $pushRows = @()
    $readbacks = [Collections.Generic.List[string]]::new()
    $i = 0
    foreach ($row in $data.files) {
        # Explicit review/source/metadata lists only; never bulk-transfer research payloads.
        if ($row.path -match '[*?\[\]]|(?i:private_custody|(^|/)(owners|qrels|labels)([./]|$)|\.(pt|bin|npy|npz|safetensors|sql|jsonl)$)') {
            throw "Research payload excluded: $($row.path)"
        }
        $local = (Get-Target $row.path).Replace('\','/')
        $remote = $RemoteRoot.TrimEnd('/') + '/' + $row.path
        $back = (Join-Path $temp "$i.readback").Replace('\','/')
        if ($Direction -eq 'Push') {
            Assert-File $row $local
            $relative = $local.Substring($rootPrefix.Length)
            $pushRows += @{ path=$relative; bytes=$row.bytes; sha256=$row.sha256; previous=$previous[$row.path] }
        }
        $readbacks.Add('get "' + $remote + '" "' + $back + '"')
        $i++
    }
    if ($Direction -eq 'Push') {
        $prepared = Invoke-PushStep 'prepare' @()
        $pending = @($prepared.pending)
        $actions = @('skip') * $data.files.Count
        if ($pending.Count) {
            try {
                $uploads = foreach ($i in $pending) {
                    $row = $data.files[$i]
                    $local = (Get-Target $row.path).Replace('\','/')
                    $remote = $RemoteRoot.TrimEnd('/') + '/' + $row.path
                    $stage = $remote.Substring(0,$remote.LastIndexOf('/')) + "/.research-sync-$token/$i"
                    'put "' + $local + '" "' + $stage + '"'
                }
                Invoke-SftpBatch $uploads
                $actions = (Invoke-PushStep 'publish' $pending).actions
            }
            catch {
                $cause = $_
                try { Invoke-PushStep 'discard' $pending | Out-Null }
                catch { Write-Warning "Check remote staging for transfer $token. $_" }
                throw $cause
            }
        }
    }
    Invoke-SftpBatch $readbacks
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
    $result = [ordered]@{ verified_at=(Get-Date -Format o); transport='SFTP'; direction=$Direction;
        remote_root=$RemoteRoot; files=$data.files; count=@($data.files).Count;
        all_size_sha256_match=$true; verification='SFTP readback size/SHA' }
    if ($Direction -eq 'Push') { $result.actions = @($actions) }
    $result | ConvertTo-Json -Depth 8
}
finally {
    # Only this invocation's unique local directory; resolved boundary checked before recursive cleanup.
    $resolved = [IO.Path]::GetFullPath($temp)
    if ([IO.Path]::GetDirectoryName($resolved) -ne $rootPath -or [IO.Path]::GetFileName($resolved) -notlike '.research-sync-*') {
        throw 'Unsafe temporary cleanup target'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
