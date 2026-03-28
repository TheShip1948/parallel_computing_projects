@echo off

:: Set build directory
set BUILD_DIR=build
set EXECUTABLE=test5.exe

:: Create the build directory if it doesn't exist
if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

:: Locate Visual Studio to load environment variables
echo Found Visual Studio, initializing environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Attempt to compile the code
echo Compiling test5.cpp...
nvcc -O2 -I..\..\eigen-3.4.0 -I..\..\include -allow-unsupported-compiler .\test5.cpp ..\..\utils\mnist_loader.cpp -o %BUILD_DIR%\%EXECUTABLE%

:: Check if the compilation succeeded
if %errorlevel% equ 0 (
    echo.
    echo Compilation Successful!
    echo Executable is located at: %BUILD_DIR%\%EXECUTABLE%
    echo Running the executable...
    %BUILD_DIR%\%EXECUTABLE%
) else (
    echo.
    echo Compilation Failed!
)
