# Windows host CPU ceiling. All launches share the same bounded CPU set.
# Existing processes survive closing this script; no kill-on-close flag is set.
# Launch this parent hidden; Windows PowerShell 5.1 children inherit its console.
[CmdletBinding()]
param(
    [string]$ConfigPath,
    [int]$AttachRootId = 0,
    [string]$Executable,
    [string[]]$CommandArguments = @()
)
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('ConfigPath')) {
    $ConfigPath = Join-Path $PSScriptRoot '../configs/runtime/local_cpu.json'
}
if ($env:OS -ne 'Windows_NT') { throw 'This launcher requires Windows Job Objects.' }
if (($AttachRootId -gt 0) -eq (-not [string]::IsNullOrWhiteSpace($Executable))) {
    throw 'Specify exactly one of AttachRootId or Executable.'
}
$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$keys = @($config.PSObject.Properties.Name | Sort-Object)
if (($keys -join ',') -ne 'cpuPercent,schemaVersion' -or
    ($config.schemaVersion -isnot [int] -and $config.schemaVersion -isnot [long]) -or $config.schemaVersion -ne 1 -or
    ($config.cpuPercent -isnot [int] -and $config.cpuPercent -isnot [long]) -or $config.cpuPercent -lt 1 -or $config.cpuPercent -gt 25) {
    throw 'Expected schemaVersion: 1 and integer cpuPercent: 1..25; unknown fields are forbidden.'
}
$checkout = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$logicalCpus = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$allowedCpus = [int][Math]::Floor($logicalCpus * $config.cpuPercent / 100)
if ($logicalCpus -gt 64 -or $allowedCpus -lt 1) {
    throw 'The aggregate ceiling cannot be represented by one Windows processor group on this host; refusing uncapped work.'
}
$affinity = [uint64]([Math]::Pow(2, $allowedCpus) - 1)
if (([uint64][Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity.ToInt64() -band $affinity) -ne $affinity) {
    throw 'Parent affinity excludes the shared CPU set; refusing an alternate uncapped CPU set.'
}
# Bound the launcher bootstrap too, before Add-Type can invoke a compiler.
[Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]([int64]$affinity)
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
if (-not ('BalootCpuJob' -as [type])) {
    # Test runtimes can remove an inherited temp directory before Add-Type runs.
    $compileTemp = [IO.Path]::GetTempPath()
    [void][IO.Directory]::CreateDirectory($compileTemp)
    $env:TEMP = $compileTemp
    $env:TMP = $compileTemp
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class BalootCpuJob {
    [StructLayout(LayoutKind.Sequential)] public struct Rate { public uint Flags, CpuRate; }
    [StructLayout(LayoutKind.Sequential)] public struct Limits {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWorkingSet, MaxWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetInformationJobObject(IntPtr job, int kind, ref Limits limits, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool QueryInformationJobObject(IntPtr job, int kind, out Rate rate, uint size, out uint length);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int id);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool member);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
    public static void Check(bool success) {
        if (!success) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    public static void Attach(IntPtr job, int id) {
        IntPtr process = OpenProcess(0x1101, false, id); // query, set quota, terminate rights
        if (process == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error(), "OpenProcess " + id);
        try {
            bool member;
            Check(IsProcessInJob(process, job, out member));
            if (!member && !AssignProcessToJobObject(job, process))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "AssignProcessToJobObject " + id);
            Check(IsProcessInJob(process, job, out member));
            if (!member) throw new InvalidOperationException("CPU job membership verification failed.");
        } finally { CloseHandle(process); }
    }
}
'@
}
$sha = [Security.Cryptography.SHA256]::Create()
try { $identity = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($checkout))).Replace('-', '') }
finally { $sha.Dispose() }
# Different Codex shells can already belong to incompatible nested jobs. Separate
# jobs use the SAME enforced affinity, so their aggregate cannot exceed the cap.
$targetId = if ($AttachRootId -gt 0) { $AttachRootId } else { $PID }
$jobName = 'Local\BalootX.Cpu.' + $identity + '.' + $targetId
$job = [BalootCpuJob]::CreateJobObject([IntPtr]::Zero, $jobName)
[BalootCpuJob]::Check($job -ne [IntPtr]::Zero)
try {
    $limits = New-Object BalootCpuJob+Limits
    $limits.Flags = 48 # AFFINITY | PRIORITY_CLASS; child overrides cannot escape.
    $limits.Affinity = [UIntPtr]$affinity
    $limits.Priority = 16384 # BELOW_NORMAL_PRIORITY_CLASS
    [BalootCpuJob]::Check([BalootCpuJob]::SetInformationJobObject($job, 2, [ref]$limits, [uint32][Runtime.InteropServices.Marshal]::SizeOf($limits)))
    # The shared job-enforced CPU set already caps aggregate host usage. Nested
    # percentage quotas multiply against their parent's quota; do not add one.
    $verified = New-Object BalootCpuJob+Rate
    $returned = [uint32]0
    [BalootCpuJob]::Check([BalootCpuJob]::QueryInformationJobObject($job, 15, [ref]$verified, 8, [ref]$returned))
    if ($returned -ne 8 -or $verified.Flags -ne 0 -or $verified.CpuRate -ne 0) {
        throw 'An older percentage-quota job exists; restart owned work before applying this CPU-set ceiling.'
    }
    if ($AttachRootId -gt 0) {
        $root = Get-Process -Id $AttachRootId
        $processes = @(Get-CimInstance Win32_Process)
        $ids = [Collections.Generic.HashSet[int]]::new()
        [void]$ids.Add($root.Id)
        do {
            $added = $false
            foreach ($row in $processes) {
                if ($ids.Contains([int]$row.ParentProcessId) -and $ids.Add([int]$row.ProcessId)) { $added = $true }
            }
        } while ($added)
        foreach ($id in $ids) {
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
                [BalootCpuJob]::Attach($job, $id)
                if ((Get-Process -Id $id).ProcessorAffinity.ToInt64() -ne $affinity) {
                    throw "Shared CPU-set verification failed for process $id."
                }
            }
        }
        Write-Output "Verified CPU ceiling: $($config.cpuPercent)%; shared CPU set: $allowedCpus/$logicalCpus; attached tree: $($ids.Count) processes."
    } else {
        [BalootCpuJob]::Attach($job, $PID)
        if ([Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity.ToInt64() -ne $affinity) {
            throw 'Shared CPU-set verification failed; command was not launched.'
        }
        [Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
        $env:BALOOT_CPU_JOB_NAME = $jobName
        Write-Output "Verified CPU ceiling: $($config.cpuPercent)%; shared CPU set: $allowedCpus/$logicalCpus; children inherit it."
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = $Executable
        $start.UseShellExecute = $false
        $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        if ($start.PSObject.Properties.Name -contains 'ArgumentList') {
            foreach ($argument in $CommandArguments) { $start.ArgumentList.Add($argument) }
        } else {
            # Windows PowerShell 5.1 has no ArgumentList; preserve native argv quotes.
            $start.Arguments = (@($CommandArguments | ForEach-Object {
                '"' + ([regex]::Replace($_, '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
            }) -join ' ')
        }
        $command = [Diagnostics.Process]::Start($start)
        try {
            $stdoutTask = $command.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
            $stderrTask = $command.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
            $command.WaitForExit()
            [void]$stdoutTask.GetAwaiter().GetResult()
            [void]$stderrTask.GetAwaiter().GetResult()
            if ($command.ExitCode -ne 0) { throw "Capped command exited with code $($command.ExitCode)." }
        } finally { $command.Dispose() }
    }
} finally { [void][BalootCpuJob]::CloseHandle($job) }
