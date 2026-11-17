[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$ArchiveSha256,
    [Parameter(Mandatory = $true)][string]$Report,
    [Parameter(Mandatory = $true)][string]$Stdout,
    [Parameter(Mandatory = $true)][string]$Stderr,
    [Parameter(Mandatory = $true)][string]$Receipt,
    [Parameter(Mandatory = $true)][string]$Workspace,
    [ValidateRange(1, 2147483647)][int64]$AffinityMask = 3,
    [ValidateRange(1, 64)][int]$NativeThreadLimit = 1,
    [string]$TaskName,
    [string]$PythonExecutable,
    [switch]$Worker
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# Scheduled workers do not inherit the interactive launcher's job. Cap this
# process before archive hashing, then inherit the same cap in every child.
$cpuCeilingReceipt = & (Join-Path $PSScriptRoot 'run_local_capped.ps1') -AttachRootId $PID
Write-Verbose ($cpuCeilingReceipt -join "`n")
$AffinityMask = $AffinityMask -band [Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity.ToInt64()
if ($AffinityMask -eq 0) { throw 'Requested affinity excludes the configured CPU set.' }

function Get-AbsolutePath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Value))
}

function Get-Sha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '')
    }
    finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function ConvertTo-PowerShellLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function ConvertTo-WindowsProcessArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes += 1
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($backslashes * 2) + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

$Archive = Get-AbsolutePath $Archive
$Report = Get-AbsolutePath $Report
$Stdout = Get-AbsolutePath $Stdout
$Stderr = Get-AbsolutePath $Stderr
$Receipt = Get-AbsolutePath $Receipt
$Workspace = Get-AbsolutePath $Workspace
$ArchiveSha256 = $ArchiveSha256.ToUpperInvariant()
if (-not $TaskName) {
    $TaskName = "BalootX-ArchiveVerify-$($ArchiveSha256.Substring(0, 16))"
}
if (-not $PythonExecutable) {
    $PythonExecutable = (
        Get-Command python.exe -CommandType Application -ErrorAction Stop |
            Select-Object -First 1
    ).Path
}
$PythonExecutable = [IO.Path]::GetFullPath($PythonExecutable)

function Invoke-ArchiveVerificationWorker {
    $mutex = [Threading.Mutex]::new($false, "Local\BalootX-ArchiveVerify-$ArchiveSha256")
    $ownsMutex = $false
    $verifier = $null
    $started = [DateTimeOffset]::Now
    $finished = $null
    $exitCode = 1
    $launcherError = $null
    $reportStatus = $null
    $actualSha256 = $null
    $cleanupErrors = [Collections.Generic.List[string]]::new()
    try {
        $ownsMutex = $mutex.WaitOne(0)
        if (-not $ownsMutex) {
            throw "Archive verification is already running for $ArchiveSha256"
        }
        foreach ($path in @($Report, $Stdout, $Stderr, $Receipt)) {
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
        }
        if (-not [IO.File]::Exists($Archive)) {
            throw "Archive does not exist: $Archive"
        }
        $actualSha256 = Get-Sha256 $Archive
        if ($actualSha256 -ne $ArchiveSha256) {
            throw "Archive SHA-256 mismatch: expected $ArchiveSha256, got $actualSha256"
        }
        foreach ($path in @($Report, $Stdout, $Stderr, $Receipt)) {
            if ([IO.File]::Exists($path)) {
                throw "Refusing to overwrite verifier evidence: $path"
            }
        }
        [IO.Directory]::CreateDirectory($Workspace) | Out-Null
        [Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]$AffinityMask
        foreach ($name in @(
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'BLIS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'ORTOOLS_NUM_THREADS'
        )) {
            [Environment]::SetEnvironmentVariable(
                $name,
                [string]$NativeThreadLimit,
                [EnvironmentVariableTarget]::Process
            )
        }
        $arguments = @(
            '-B',
            'scripts/verify_archive.py',
            $Archive,
            '--output',
            $Report,
            '--workspace',
            $Workspace,
            '--keep'
        )
        $processInfo = [Diagnostics.ProcessStartInfo]::new()
        $processInfo.FileName = $PythonExecutable
        $processInfo.Arguments = (($arguments | ForEach-Object {
            ConvertTo-WindowsProcessArgument ([string]$_)
        }) -join ' ')
        $processInfo.WorkingDirectory = $RepositoryRoot
        $processInfo.UseShellExecute = $false
        $processInfo.CreateNoWindow = $true
        $processInfo.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
        $processInfo.RedirectStandardOutput = $true
        $processInfo.RedirectStandardError = $true
        $processInfo.StandardOutputEncoding = [Text.UTF8Encoding]::new($false)
        $processInfo.StandardErrorEncoding = [Text.UTF8Encoding]::new($false)
        $verifier = [Diagnostics.Process]::new()
        $verifier.StartInfo = $processInfo
        if (-not $verifier.Start()) {
            throw 'The archive verifier process did not start.'
        }
        $verifier.ProcessorAffinity = [IntPtr]$AffinityMask
        $stdoutTask = $verifier.StandardOutput.ReadToEndAsync()
        $stderrTask = $verifier.StandardError.ReadToEndAsync()
        $verifier.WaitForExit()
        [IO.File]::WriteAllText($Stdout, $stdoutTask.Result, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($Stderr, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
        $exitCode = [int]$verifier.ExitCode
        if ([IO.File]::Exists($Report)) {
            try {
                $reportStatus = (ConvertFrom-Json ([IO.File]::ReadAllText($Report))).status
            }
            catch {
                $launcherError = "Invalid verifier report: $($_.Exception.Message)"
                $exitCode = 1
            }
        }
        if ($exitCode -eq 0 -and $reportStatus -ne 'passed') {
            $launcherError = 'Verifier exited successfully without a passed report.'
            $exitCode = 1
        }
    }
    catch {
        $launcherError = $_.Exception.ToString()
        $exitCode = 1
    }
    finally {
        if ($null -ne $verifier -and -not $verifier.HasExited) {
            try {
                $terminated = Start-Process `
                    -FilePath (Join-Path $env:SystemRoot 'System32\taskkill.exe') `
                    -ArgumentList @('/PID', [string]$verifier.Id, '/T', '/F') `
                    -WindowStyle Hidden `
                    -Wait `
                    -PassThru
                if ($terminated.ExitCode -ne 0) {
                    $cleanupErrors.Add("taskkill_return_code:$($terminated.ExitCode)")
                }
                $verifier.WaitForExit()
            }
            catch {
                $cleanupErrors.Add("taskkill:$($_.Exception.GetType().Name)")
            }
        }
        $finished = [DateTimeOffset]::Now
        if (-not [IO.File]::Exists($Receipt)) {
            $receiptData = [ordered]@{
                schemaVersion = 1
                taskName = $TaskName
                archive = $Archive
                expectedArchiveSha256 = $ArchiveSha256
                actualArchiveSha256 = $actualSha256
                workerPid = $PID
                verifierPid = if ($null -ne $verifier) { $verifier.Id } else { $null }
                affinityMask = ('0x{0:X}' -f $AffinityMask)
                cpuCeiling = $cpuCeilingReceipt
                nativeThreadLimit = $NativeThreadLimit
                startedAt = $started.ToString('o')
                finishedAt = $finished.ToString('o')
                exitCode = $exitCode
                status = if ($exitCode -eq 0 -and $reportStatus -eq 'passed') { 'passed' } else { 'failed' }
                reportStatus = $reportStatus
                report = $Report
                stdout = $Stdout
                stderr = $Stderr
                workspace = $Workspace
                launcherError = $launcherError
                cleanupErrors = @($cleanupErrors)
            }
            $temporaryReceipt = "$Receipt.$PID.tmp"
            $json = ($receiptData | ConvertTo-Json -Depth 5) + "`n"
            [IO.File]::WriteAllText(
                $temporaryReceipt,
                $json,
                [Text.UTF8Encoding]::new($false)
            )
            [IO.File]::Move($temporaryReceipt, $Receipt)
        }
        if ($ownsMutex) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
        try {
            Start-Process `
                -FilePath (Join-Path $env:SystemRoot 'System32\schtasks.exe') `
                -ArgumentList @('/Delete', '/TN', $TaskName, '/F') `
                -WindowStyle Hidden `
                -Wait | Out-Null
        }
        catch {
            # A completed receipt remains authoritative if task cleanup is unavailable.
        }
    }
    exit $exitCode
}

if ($Worker) {
    Invoke-ArchiveVerificationWorker
}

if (-not [IO.File]::Exists($Archive)) {
    throw "Archive does not exist: $Archive"
}
$actualSha256 = Get-Sha256 $Archive
if ($actualSha256 -ne $ArchiveSha256) {
    throw "Archive SHA-256 mismatch: expected $ArchiveSha256, got $actualSha256"
}
foreach ($path in @($Report, $Stdout, $Stderr, $Receipt)) {
    if ([IO.File]::Exists($path)) {
        throw "Refusing to overwrite verifier evidence: $path"
    }
}
if ([IO.Directory]::Exists($Workspace)) {
    throw "Refusing to reuse verifier workspace: $Workspace"
}

Import-Module ScheduledTasks -ErrorAction Stop
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Archive verification task already exists: $TaskName"
}

$scriptPath = [IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)
$workerCommand = @(
    '&',
    (ConvertTo-PowerShellLiteral $scriptPath),
    '-Worker',
    '-Archive',
    (ConvertTo-PowerShellLiteral $Archive),
    '-ArchiveSha256',
    (ConvertTo-PowerShellLiteral $ArchiveSha256),
    '-Report',
    (ConvertTo-PowerShellLiteral $Report),
    '-Stdout',
    (ConvertTo-PowerShellLiteral $Stdout),
    '-Stderr',
    (ConvertTo-PowerShellLiteral $Stderr),
    '-Receipt',
    (ConvertTo-PowerShellLiteral $Receipt),
    '-Workspace',
    (ConvertTo-PowerShellLiteral $Workspace),
    '-AffinityMask',
    [string]$AffinityMask,
    '-NativeThreadLimit',
    [string]$NativeThreadLimit,
    '-TaskName',
    (ConvertTo-PowerShellLiteral $TaskName),
    '-PythonExecutable',
    (ConvertTo-PowerShellLiteral $PythonExecutable)
) -join ' '
$encodedWorker = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes($workerCommand)
)
$windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$action = New-ScheduledTaskAction `
    -Execute $windowsPowerShell `
    -Argument "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -EncodedCommand $encodedWorker" `
    -WorkingDirectory $RepositoryRoot
$trigger = New-ScheduledTaskTrigger -Once -At ([DateTime]::Now.AddMinutes(1))
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -DisallowHardTerminate `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal `
    -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $TaskName

[ordered]@{
    schemaVersion = 1
    status = 'scheduled'
    taskName = $TaskName
    archive = $Archive
    archiveSha256 = $ArchiveSha256
    report = $Report
    receipt = $Receipt
    workspace = $Workspace
} | ConvertTo-Json -Depth 3
