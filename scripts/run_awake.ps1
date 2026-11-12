<#
Run a WSL script while keeping Windows from going to sleep, e.g. a long training run on a laptop.
Only the system stays awake: the display still turns off and the screen still locks on their usual timeouts.
Closing the lid still sleeps the laptop.
    powershell -ExecutionPolicy Bypass -File scripts\run_awake.ps1 /home/me/train.sh [-Distro Ubuntu-24.04]
#>
param([Parameter(Mandatory = $true)][string]$Script, [string]$Distro = "Ubuntu-24.04")

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED, without ES_DISPLAY_REQUIRED. Windows drops the request when this process ends.
if ([Win32.Power]::SetThreadExecutionState([uint32]2147483648 -bor [uint32]1) -eq 0) {
    Write-Warning "Windows refused the keep-awake request; the laptop may sleep during the run."
}
& wsl.exe -d $Distro -- bash -l $Script
exit $LASTEXITCODE
