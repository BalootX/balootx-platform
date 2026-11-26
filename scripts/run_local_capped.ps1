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

function Get-Descendants($Root, $Processes) {
    $snapshotRoot = $Processes | Where-Object ProcessId -eq $Root.ProcessId | Select-Object -First 1
    if ($snapshotRoot.CreationDate -ne $Root.CreationDate) {
        throw "Root process $($Root.ProcessId) exited during CPU tree attachment."
    }
    $children = $Processes | Group-Object ParentProcessId -AsHashTable -AsString
    $parents = [Collections.Generic.Queue[object]]::new()
    $parents.Enqueue($Root)
    while ($parents.Count) {
        $parent = $parents.Dequeue()
        foreach ($child in $children[[string]$parent.ProcessId].Group) {
            if ($child.ProcessId -eq $Root.ProcessId) { continue }
            if ($null -eq $child.CreationDate) {
                throw "Cannot verify creation time for process $($child.ProcessId) ($($child.Name), parent $($child.ParentProcessId))."
            }
            if ($child.CreationDate.ToUniversalTime() -ge $parent.CreationDate.ToUniversalTime()) {
                $child
                $parents.Enqueue($child)
            }
        }
    }
}

function Confirm-CappedProcess($Process, [bool]$VerifyOnly = $false) {
    $created = $Process.CreationDate.ToUniversalTime()
    $identity = "$($Process.ProcessId) ($($Process.Name), parent $($Process.ParentProcessId), created $($created.ToString('o')))"
    [BalootCpuJob]::Attach($job, $Process.ProcessId, $created.ToFileTimeUtc(), $affinity, $identity, $VerifyOnly)
}

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
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool GetProcessTimes(IntPtr process, out long created, out long exited, out long kernel, out long user);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool GetProcessAffinityMask(IntPtr process, out UIntPtr affinity, out UIntPtr systemAffinity);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool member);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
    public static void Check(bool success) {
        if (!success) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    private static Win32Exception ProcessError(int code, string operation, string identity) {
        return new Win32Exception(code, operation + " (Win32 " + code + ") for process " + identity
            + ": " + new Win32Exception(code).Message);
    }
    private static void CheckProcess(bool success, string operation, string identity) {
        if (!success) throw ProcessError(Marshal.GetLastWin32Error(), operation, identity);
    }
    private static bool HasExited(IntPtr process, string identity) {
        uint state = WaitForSingleObject(process, 0);
        CheckProcess(state != 0xFFFFFFFF, "WaitForSingleObject", identity);
        return state == 0; // WAIT_OBJECT_0
    }
    public static bool Attach(IntPtr job, int id, long expectedCreation, ulong affinity, string identity, bool verifyOnly) {
        const int InvalidParameter = 87, AccessDenied = 5;
        const uint QueryAndSynchronize = 0x101000, SetQuotaAndTerminate = 0x0101;
        uint rights = verifyOnly ? QueryAndSynchronize : QueryAndSynchronize | SetQuotaAndTerminate;
        IntPtr process = OpenProcess(rights, false, id);
        if (process == IntPtr.Zero) {
            int error = Marshal.GetLastWin32Error();
            if (error == InvalidParameter) return false; // No process for this positive PID.
            if (error == AccessDenied && !verifyOnly) {
                // A retained, exited process can still have a PID. Probe with query rights only.
                IntPtr status = OpenProcess(QueryAndSynchronize, false, id);
                if (status != IntPtr.Zero) {
                    try { if (HasExited(status, identity)) return false; }
                    finally { CloseHandle(status); }
                } else if (Marshal.GetLastWin32Error() == InvalidParameter) {
                    return false;
                }
            }
            throw ProcessError(error, "OpenProcess", identity);
        }
        try {
            if (HasExited(process, identity)) return false;
            bool member;
            UIntPtr actualAffinity, systemAffinity;
            try {
                if (expectedCreation != 0) {
                    long created, exited, kernel, user;
                    CheckProcess(GetProcessTimes(process, out created, out exited, out kernel, out user), "GetProcessTimes", identity);
                    // CIM timestamps have microsecond precision. The handle pins the current process.
                    if (created / 10 > expectedCreation / 10) return false;
                    if (created / 10 != expectedCreation / 10)
                        throw new InvalidOperationException("Process creation time verification failed for " + identity + ".");
                }
                CheckProcess(IsProcessInJob(process, job, out member), "IsProcessInJob", identity);
                if (!member && !verifyOnly) {
                    CheckProcess(AssignProcessToJobObject(job, process), "AssignProcessToJobObject", identity);
                    CheckProcess(IsProcessInJob(process, job, out member), "IsProcessInJob", identity);
                }
                CheckProcess(GetProcessAffinityMask(process, out actualAffinity, out systemAffinity), "GetProcessAffinityMask", identity);
            } catch (Win32Exception) {
                if (HasExited(process, identity)) return false;
                throw;
            }
            if (HasExited(process, identity)) return false;
            if (!member) throw new InvalidOperationException("CPU job membership verification failed for " + identity + ".");
            if (actualAffinity.ToUInt64() != affinity)
                throw new InvalidOperationException("Shared CPU-set verification failed for " + identity + ".");
            return true;
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
        $properties = 'ProcessId', 'ParentProcessId', 'CreationDate', 'Name'
        $root = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $AttachRootId" -Property $properties
        if ($null -eq $root.CreationDate) { throw "Cannot verify creation time for root process $AttachRootId." }
        if (-not (Confirm-CappedProcess $root)) { throw "Root process $AttachRootId exited before its CPU cap was verified." }
        $verifiedTree = @{$AttachRootId = $root.CreationDate}
        $processes = Get-CimInstance -ClassName Win32_Process -Property $properties
        foreach ($process in (Get-Descendants $root $processes)) {
            if (Confirm-CappedProcess $process) { $verifiedTree[[int]$process.ProcessId] = $process.CreationDate }
        }
        # A descendant can spawn children before it joins the job. Check once after all assignments.
        $processes = Get-CimInstance -ClassName Win32_Process -Property $properties
        foreach ($process in (Get-Descendants $root $processes)) {
            $id = [int]$process.ProcessId
            if ($verifiedTree[$id] -ne $process.CreationDate -and (Confirm-CappedProcess $process $true)) {
                $verifiedTree[$id] = $process.CreationDate
            }
        }
        Write-Output "Verified CPU ceiling: $($config.cpuPercent)%; shared CPU set: $allowedCpus/$logicalCpus; attached tree: $($verifiedTree.Count) processes."
    } else {
        if (-not [BalootCpuJob]::Attach($job, $PID, 0, $affinity, "$PID (launcher)", $false)) {
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
