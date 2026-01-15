@echo off
setlocal
echo Setting up VS environment...
call "D:\ProgramInstallation\Microsoft_build_tools_2022_17.14\VC\Auxiliary\Build\vcvars64.bat"

echo Cleaning previous build...
if exist build rmdir /s /q build
mkdir build

echo Configuring with CMake...
cmake -G "Ninja" -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native

if %errorlevel% neq 0 (
    echo CMake Configuration failed!
    exit /b %errorlevel%
)

echo Building...
cmake --build build

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo Build successful! Executable is: build\vector_multiply.exe
endlocal
