@echo off
setlocal
echo Setting up VS environment...
call "D:\ProgramInstallation\Microsoft_build_tools_2022_17.14\VC\Auxiliary\Build\vcvars64.bat"

echo Cleaning previous build...
if exist build rmdir /s /q build
mkdir build

echo Configuring with CMake (Ninja)...
cmake -G "Ninja" -S . -B build -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo CMake Configuration failed!
    echo Attempting fallback to direct NVCC compilation...
    
    echo Building with NVCC...
    nvcc main.cu -o build/convolution.exe -Xcompiler /MD
    
    if %errorlevel% neq 0 (
        echo Fallback Build failed!
        exit /b %errorlevel%
    )
    echo Fallback Build Successful! Executable is in build/convolution.exe
    exit /b 0
)

echo Building...
cmake --build build

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo Build successful! Executable is: build\convolution.exe
endlocal
