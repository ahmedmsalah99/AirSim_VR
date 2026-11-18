@echo off
setlocal
REM Generates Visual Studio project files with the engine named by the .uproject's
REM EngineAssociation, by calling that engine's own UnrealBuildTool.
REM We don't go through the UnrealVersionSelector registered for .uproject files:
REM the launcher's copy is updated for newer engines and looks for UnrealBuildTool
REM at Engine\Binaries\DotNET\UnrealBuildTool\, which doesn't exist in UE 4.24.

set "UPROJECT="
for %%f in (*.uproject) do set "UPROJECT=%%~ff"
if not defined UPROJECT (
	echo No .uproject file found in %cd%
	exit /b 1
)
echo Generating files for %UPROJECT%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
	"$p = '%UPROJECT%';" ^
	"$a = (Get-Content -Raw $p | ConvertFrom-Json).EngineAssociation;" ^
	"if ($a -like '{*') { $d = (Get-ItemProperty 'HKCU:\Software\Epic Games\Unreal Engine\Builds' -ErrorAction SilentlyContinue).$a }" ^
	"else {" ^
	"  foreach ($k in 'HKLM:\SOFTWARE\EpicGames\Unreal Engine\', 'HKLM:\SOFTWARE\WOW6432Node\EpicGames\Unreal Engine\') { if (-not $d) { $d = (Get-ItemProperty ($k + $a) -ErrorAction SilentlyContinue).InstalledDirectory } };" ^
	"  $dat = Join-Path $env:ProgramData 'Epic\UnrealEngineLauncher\LauncherInstalled.dat';" ^
	"  if (-not $d -and (Test-Path $dat)) { $d = ((Get-Content -Raw $dat | ConvertFrom-Json).InstallationList | Where-Object { $_.AppName -eq ('UE_' + $a) } | Select-Object -First 1).InstallLocation };" ^
	"  $def = Join-Path $env:ProgramFiles ('Epic Games\UE_' + $a);" ^
	"  if (-not $d -and (Test-Path $def)) { $d = $def }" ^
	"};" ^
	"if (-not $d) { Write-Host ('Could not find where engine ' + $a + ' is installed'); exit 1 };" ^
	"$ubt = @('Engine\Binaries\DotNET\UnrealBuildTool.exe', 'Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.exe') | ForEach-Object { Join-Path $d $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1;" ^
	"if (-not $ubt) { Write-Host ('UnrealBuildTool.exe not found under ' + $d); exit 1 };" ^
	"$kind = if (Test-Path (Join-Path $d 'Engine\Build\InstalledBuild.txt')) { '-rocket' } else { '-engine' };" ^
	"Write-Host ('Using ' + $ubt);" ^
	"& $ubt -projectfiles ('-project=' + $p) -game $kind -progress;" ^
	"exit $LASTEXITCODE"
if ERRORLEVEL 1 (
	echo Failed to generate project files.
	exit /b 1
)
