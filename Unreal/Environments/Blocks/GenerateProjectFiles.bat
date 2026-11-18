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
	"else { $d = (Get-ItemProperty ('HKLM:\SOFTWARE\EpicGames\Unreal Engine\' + $a) -ErrorAction SilentlyContinue).InstalledDirectory };" ^
	"if (-not $d) { Write-Host ('Engine ' + $a + ' is not registered on this machine'); exit 1 };" ^
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
