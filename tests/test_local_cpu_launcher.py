"""One Windows acceptance check for the operational process-tree CPU ceiling."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_launcher_restores_missing_add_type_temp_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "removed" / "compiler"
    env = {**os.environ, "TEMP": str(missing), "TMP": str(missing)}
    command = (f"& '{root / 'scripts/run_local_capped.ps1'}' -Executable '{sys.executable}' "
               "-CommandArguments @('-X','utf8','-c','from baloot_platform.config import platform_root; platform_root(); print(123)')")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        env=env, text=True, capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr
    assert missing.is_dir()
    assert "Verified CPU ceiling:" in result.stdout and "123" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
@pytest.mark.parametrize("refused", [False, True])
def test_direct_python_runtime_enforces_cap_once_and_fails_closed(monkeypatch, refused):
    from baloot_platform.config import _enforce_local_cpu_ceiling, platform_root

    launch = Mock(
        side_effect=subprocess.CalledProcessError(1, "cpu cap") if refused else None
    )
    monkeypatch.delenv("BALOOT_CPU_JOB_NAME", raising=False)
    monkeypatch.setattr(subprocess, "run", launch)
    _enforce_local_cpu_ceiling.cache_clear()
    try:
        if refused:
            with pytest.raises(subprocess.CalledProcessError):
                platform_root()
        else:
            platform_root()
            platform_root()
        assert launch.call_count == 1
        command = launch.call_args.args[0]
        assert Path(command[command.index("-File") + 1]).name == "run_local_capped.ps1"
        assert "-NonInteractive" in command
        assert command[command.index("-WindowStyle") + 1] == "Hidden"
        assert command[-2:] == ["-AttachRootId", str(os.getpid())]
        assert launch.call_args.kwargs["check"] is True
        assert launch.call_args.kwargs["creationflags"] == subprocess.CREATE_NEW_CONSOLE
        startup = launch.call_args.kwargs["startupinfo"]
        assert startup.dwFlags & subprocess.STARTF_USESHOWWINDOW and startup.wShowWindow == 0
    finally:
        _enforce_local_cpu_ceiling.cache_clear()


@pytest.mark.parametrize("invalid", [
    "no_env", "missing_job", "not_member", "query_failure", "rate", "flags",
    "affinity", "priority", "breakaway", "process_affinity", "config_changed", "unknown_field",
    "coerced_percent", "coerced_schema", "invalid_schema", "out_of_range",
    "invalid_json", "too_many_cpus", "too_few_cpus", None,
])
def test_inherited_cpu_ceiling_requires_live_kernel_limits(monkeypatch, tmp_path, invalid):
    import ctypes
    from baloot_platform.config import _verified_inherited_cpu_ceiling

    config_dir = tmp_path / "configs/runtime"
    config_dir.mkdir(parents=True)
    config = {"schemaVersion": 1, "cpuPercent": 5}
    if invalid == "config_changed":
        config["cpuPercent"] = 10
    elif invalid == "unknown_field":
        config["unknown"] = 0
    elif invalid == "coerced_percent":
        config["cpuPercent"] = 5.0
    elif invalid == "coerced_schema":
        config["schemaVersion"] = True
    elif invalid == "invalid_schema":
        config["schemaVersion"] = 2
    elif invalid == "out_of_range":
        config["cpuPercent"] = 26
    (config_dir / "local_cpu.json").write_text(
        "invalid" if invalid == "invalid_json" else json.dumps(config)
    )
    monkeypatch.setenv("BALOOT_CPU_JOB_NAME", "Local\\BalootX.Cpu.test")
    if invalid == "no_env":
        monkeypatch.delenv("BALOOT_CPU_JOB_NAME")
    monkeypatch.setattr(os, "cpu_count", lambda: 128 if invalid == "too_many_cpus" else 1 if invalid == "too_few_cpus" else 24)

    def member(_process, _job, value):
        value._obj.value = invalid != "not_member"
        return True

    def query(_job, kind, value, _size, _returned):
        if invalid == "query_failure":
            return False
        if kind == 15:
            value._obj[0] = 1 if invalid == "rate" else 0
            value._obj[1] = 0
        else:
            value._obj.flags = 16 if invalid == "flags" else 48 | 0x800 if invalid == "breakaway" else 48
            value._obj.affinity = 3 if invalid == "affinity" else 1
            value._obj.priority = 32 if invalid == "priority" else 16384
        return True

    def process_affinity(_process, process, system):
        process._obj.value = 3 if invalid == "process_affinity" else 1
        system._obj.value = (1 << 24) - 1
        return True

    kernel = Mock()
    kernel.OpenJobObjectW.return_value = 0 if invalid == "missing_job" else 1
    kernel.IsProcessInJob.side_effect = member
    kernel.QueryInformationJobObject.side_effect = query
    kernel.GetProcessAffinityMask.side_effect = process_affinity
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    assert _verified_inherited_cpu_ceiling(tmp_path) is (invalid is None)
    if kernel.OpenJobObjectW.called and invalid != "missing_job":
        kernel.CloseHandle.assert_called_once_with(1)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_current_capped_process_has_verified_inherited_limits():
    from baloot_platform.config import _verified_inherited_cpu_ceiling

    if not os.environ.get("BALOOT_CPU_JOB_NAME"):
        pytest.skip("Requires the capped launcher")
    assert _verified_inherited_cpu_ceiling(Path(__file__).resolve().parents[1])


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_verified_inherited_cpu_ceiling_skips_nested_launcher(monkeypatch):
    import baloot_platform.config as config

    monkeypatch.setattr(config, "_verified_inherited_cpu_ceiling", lambda _checkout: True)
    launch = Mock()
    monkeypatch.setattr(subprocess, "run", launch)
    config._enforce_local_cpu_ceiling.cache_clear()
    try:
        config.platform_root()
        launch.assert_not_called()
    finally:
        config._enforce_local_cpu_ceiling.cache_clear()


def _assert_attach_handles_process_lifetimes(launcher):
    # Shadow CIM in this shell only. The launcher still opens and caps real processes.
    command = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FixtureDacl {
    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern bool SetKernelObjectSecurity(IntPtr handle, uint information, byte[] descriptor);
}
'@
$start = [Diagnostics.ProcessStartInfo]::new($env:BALOOT_FIXTURE_PYTHON, '-c "import sys; sys.stdin.read()"')
$start.UseShellExecute = $false
$start.RedirectStandardInput = $true
$start.CreateNoWindow = $true
$exiting = [Diagnostics.Process]::Start($start)
$denied = [Diagnostics.Process]::Start($start)
$late = [Diagnostics.Process]::Start($start)
try {
    $rootRow = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $exitingRow = Get-CimInstance Win32_Process -Filter "ProcessId = $($exiting.Id)"
    $deniedRow = Get-CimInstance Win32_Process -Filter "ProcessId = $($denied.Id)"
    $lateRow = Get-CimInstance Win32_Process -Filter "ProcessId = $($late.Id)"
    foreach ($row in @($exitingRow, $deniedRow, $lateRow)) {
        if ($row.ParentProcessId -ne $PID) { throw 'Fixture is not a child of the attach root.' }
    }
    $dacl = [Security.AccessControl.RawSecurityDescriptor]::new('D:')
    $descriptor = [byte[]]::new($dacl.BinaryLength)
    $dacl.GetBinaryForm($descriptor, 0)
    if (-not [FixtureDacl]::SetKernelObjectSecurity($denied.Handle, 4, $descriptor)) {
        throw [ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
    }
    $olderRow = [pscustomobject]@{
        ProcessId = $deniedRow.ProcessId
        ParentProcessId = $deniedRow.ParentProcessId
        Name = $deniedRow.Name
        CreationDate = $rootRow.CreationDate.AddSeconds(-1)
    }
    $fixture = @{ IncludeExitingRow = $true; ChildRow = $olderRow; ShowLateOnRead = 0; Reads = 0 }
    function Get-CimInstance {
        param([string]$ClassName, [string]$Filter, [string[]]$Property)
        if ($ClassName -ne 'Win32_Process') {
            CimCmdlets\Get-CimInstance -ClassName $ClassName
            return
        }
        $rootRow
        if ($Filter) { return }
        $fixture.Reads++
        if ($fixture.IncludeExitingRow) {
            $fixture.IncludeExitingRow = $false
            $exiting.StandardInput.Close()
            if (-not $exiting.WaitForExit(5000)) { throw 'Fixture child did not exit.' }
            $exiting.Dispose()
            $exitingRow
        }
        $fixture.ChildRow
        if ($fixture.Reads -eq $fixture.ShowLateOnRead) { $lateRow }
    }
    & $env:BALOOT_FIXTURE_LAUNCHER -AttachRootId $PID

    $fixture.ShowLateOnRead = $fixture.Reads + 2
    try {
        & $env:BALOOT_FIXTURE_LAUNCHER -AttachRootId $PID
        throw 'The readback accepted an uncapped descendant.'
    } catch {
        if ($_.Exception.Message -notmatch 'CPU job membership verification failed for') { throw }
        'Uncapped descendant detected at readback'
    }

    $fixture.ChildRow = $deniedRow
    $created = $deniedRow.CreationDate.ToUniversalTime()
    "Denied fixture: $($deniedRow.ProcessId) ($($deniedRow.Name), parent $PID, created $($created.ToString('o')))"
    & $env:BALOOT_FIXTURE_LAUNCHER -AttachRootId $PID
} finally {
    $exiting.Dispose()
    $late.StandardInput.Close()
    $late.Dispose()
    try {
        if ($denied.HasExited) { throw 'Access-denied fixture exited before the check.' }
        $denied.StandardInput.Close()
        if (-not $denied.WaitForExit(5000)) { throw 'Access-denied fixture did not exit.' }
    } finally { $denied.Dispose() }
}
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        env={
            **os.environ,
            "BALOOT_FIXTURE_LAUNCHER": str(launcher),
            "BALOOT_FIXTURE_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.stdout.count("Verified CPU ceiling:") == 1, result.stderr
    assert "attached tree: 1 processes." in result.stdout
    assert "Uncapped descendant detected at readback" in result.stdout
    assert result.returncode != 0, result
    assert "Denied fixture: " in result.stdout, result
    identity = result.stdout.split("Denied fixture: ", 1)[1].splitlines()[0]
    assert f"OpenProcess (Win32 5) for process {identity}" in " ".join(result.stderr.split())


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_local_cpu_cap_rejects_invalid_config_and_is_inherited(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/run_local_capped.ps1"
    source = launcher.read_text(encoding="utf-8")
    assert source.index(".ProcessorAffinity = [IntPtr]([int64]$affinity)") < source.index("Add-Type -TypeDefinition")
    assert "$start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden" in source
    assert "$start.CreateNoWindow = $true" not in source
    config = root / "configs/runtime/local_cpu.json"
    ceiling = json.loads(config.read_text())["cpuPercent"]
    assert type(ceiling) is int and 1 <= ceiling <= 25
    prefix = ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-File", str(launcher)]
    hidden = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=0)
    invalids = [
        {"schemaVersion": 1, "cpuPercent": value}
        for value in (0, 26, 5.0, True, "5")
    ] + [
        {"schemaVersion": 2, "cpuPercent": 5},
        {"schemaVersion": 1, "cpuPercent": 5, "unknown": 0},
    ]
    for invalid in invalids:
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(invalid))
        result = subprocess.run(
            [*prefix, "-ConfigPath", str(path), "-AttachRootId", str(os.getpid())],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=hidden,
        )
        assert result.returncode != 0 and "Expected schemaVersion" in result.stderr

    # Windows PowerShell -File must resolve the default path after param binding.
    default = subprocess.run(
        [*prefix, "-AttachRootId", str(os.getpid())],
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=hidden,
    )
    assert default.returncode == 0, default.stderr
    assert f"Verified CPU ceiling: {ceiling}%" in default.stdout

    # Verify membership, job-enforced affinity, and absence of multiplying quotas.
    probe = r'''
import ctypes, os, subprocess, sys
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.GetConsoleWindow.restype = ctypes.c_void_p
user = ctypes.WinDLL("user32", use_last_error=True)
user.IsWindowVisible.argtypes = [ctypes.c_void_p]
assert not user.IsWindowVisible(k.GetConsoleWindow()), "Capped child must not show a console window"
k.OpenJobObjectW.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_wchar_p]
k.OpenJobObjectW.restype = ctypes.c_void_p
k.GetCurrentProcess.restype = ctypes.c_void_p
k.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
k.CloseHandle.argtypes = [ctypes.c_void_p]
name = os.environ["BALOOT_CPU_JOB_NAME"]
h = k.OpenJobObjectW(4, False, name)
assert h, ctypes.get_last_error()
try:
    member = ctypes.c_bool()
    assert k.IsProcessInJob(k.GetCurrentProcess(), h, ctypes.byref(member)) and member.value
    rate = (ctypes.c_uint * 2)()
    assert k.QueryInformationJobObject(h, 15, ctypes.byref(rate), 8, None)
    ceiling = int(sys.argv[1])
    assert list(rate) == [0, 0], list(rate)
    class Limits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
            ("flags", ctypes.c_uint), ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t), ("active_processes", ctypes.c_uint),
            ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint),
            ("scheduling", ctypes.c_uint),
        ]
    limits = Limits()
    assert k.QueryInformationJobObject(h, 2, ctypes.byref(limits), ctypes.sizeof(limits), None)
    count = int(os.cpu_count() * ceiling / 100)
    assert limits.flags & 48 == 48 and limits.affinity == (1 << count) - 1
    assert limits.priority == 16384
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    assert k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask))
    assert process_mask.value == (1 << count) - 1, process_mask.value
    # Even explicitly widening affinity cannot escape the job's enforced set.
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetProcessAffinityMask(k.GetCurrentProcess(), system_mask.value)
    assert k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask))
    assert process_mask.value == (1 << count) - 1, process_mask.value
finally:
    k.CloseHandle(h)
from pathlib import Path
from unittest.mock import patch
import json
import baloot_platform.config as config
root = Path(config.__file__).resolve().parents[2]
if ceiling == json.loads((root / "configs/runtime/local_cpu.json").read_text())["cpuPercent"]:
    assert config._verified_inherited_cpu_ceiling(root)
    config._enforce_local_cpu_ceiling.cache_clear()
    with patch("subprocess.run", side_effect=AssertionError("nested CPU launcher")):
        config.platform_root()
if sys.argv[2] != "grandchild":
    subprocess.check_call([sys.executable, "-c", sys.argv[2], sys.argv[1], "grandchild"],
                          stdout=sys.stdout, stderr=sys.stderr)
print("INHERITED_HARD_CAP_OK")
print("HEADLESS_STDERR_OK", file=sys.stderr)
'''
    # Exercise configuration without editing the live server's ceiling.
    configured = tmp_path / "valid.json"
    alternate = min(25, ceiling + 1)
    if int(os.cpu_count() * alternate / 100) != int(os.cpu_count() * ceiling / 100):
        alternate = ceiling - 1
    for percent in (ceiling, alternate):
        configured.write_text(json.dumps({"schemaVersion": 1, "cpuPercent": percent}))
        command = (
            f"& '{launcher}' -ConfigPath '{configured}' -Executable '{sys.executable}' "
            "-CommandArguments @('-c', @'\n" + probe + "\n'@, '" + str(percent) + "', @'\n" + probe + "\n'@)"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command],
            capture_output=True, text=True, timeout=60,
            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=hidden,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("INHERITED_HARD_CAP_OK") == 2
        assert result.stderr.count("HEADLESS_STDERR_OK") == 2
        assert "VoidTaskResult" not in result.stdout
    _assert_attach_handles_process_lifetimes(launcher)
