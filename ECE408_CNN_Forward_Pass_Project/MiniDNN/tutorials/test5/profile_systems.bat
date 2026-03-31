@echo off
setlocal

:: Set variables
set BUILD_DIR=build
set EXECUTABLE=%BUILD_DIR%\test5.exe
set REPORT_NAME=test5_profile

:: Check if executable exists
if not exist %EXECUTABLE% (
    echo [ERROR] %EXECUTABLE% not found. 
    echo Please run build.bat first to compile the project.
    exit /b 1
)

:: Run Nsight Systems profiling
echo.
echo ===========================================================
echo Profiling %EXECUTABLE% with NVIDIA Nsight Systems (NSYS)
echo ===========================================================
echo.

:: --stats=true: provides a summary of CUDA kernels and memory transfers in the console
:: -o: specifies the output report name
:: --force-overwrite true: overwrites existing report files
nsys profile --stats=true -o %REPORT_NAME% --force-overwrite true %EXECUTABLE%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Profiling completed successfully.
    echo Report file: %REPORT_NAME%.nsys-rep
    echo.
    echo To view the detailed timeline, open the .nsys-rep file in Nsight Systems GUI.
) else (
    echo.
    echo [ERROR] Profiling failed with error code %errorlevel%.
)

endlocal
