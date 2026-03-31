@echo off
setlocal

:: Set the path for Nsight Compute 2019.4 CLI
:: Note: This version is specifically used to support Pascal architecture (GTX 10x0)
set NCU_CLI="C:\Program Files\NVIDIA Corporation\Nsight Compute 2019.4.0\target\windows-desktop-win7-x64\nv-nsight-cu-cli.exe"

set BUILD_DIR=build
set EXECUTABLE=%BUILD_DIR%\test5.exe
set REPORT_NAME=test5_compute_profile

:: Check if executable exists
if not exist %EXECUTABLE% (
    echo [ERROR] %EXECUTABLE% not found. 
    echo Please run build.bat first to compile the project.
    exit /b 1
)

:: Run Nsight Compute profiling
echo.
echo ===========================================================
echo Profiling %EXECUTABLE% with Nsight Compute 2019.4.0
echo (Pascal Architecture Support)
echo ===========================================================
echo [WARNING] Nsight Compute is slow because it replays kernels.
echo For 1000 images, this may take a few minutes.
echo.

:: -o: specifies the output report name
:: -f: force overwrite existing report
%NCU_CLI% -o %REPORT_NAME% -f %EXECUTABLE%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Profiling completed successfully.
    echo Report file: %REPORT_NAME%.nsight-cuprof-report
    echo.
    echo To view the detailed metrics, open the report in Nsight Compute 2019.4 GUI.
) else (
    echo.
    echo [ERROR] Profiling failed with error code %errorlevel%.
)

endlocal
