@echo off
setlocal
echo Setting up VS environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Cleaning previous build...
if exist build rmdir /s /q build
mkdir build

echo Configuring with CMake (Ninja)...
cmake -G "Ninja" -S . -B build -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo CMake Configuration failed!
    echo Attempting fallback to direct NVCC compilation...
    
    echo Building with NVCC...
    nvcc main.cu -o build/scan.exe -Xcompiler /MD
    
    if not exist build/scan.exe (
        echo Fallback Build failed! Resulting executable not found.
        exit /b 1
    )
    echo Fallback Build Successful! Executable is in build/scan.exe
    exit /b 0
)

echo Building...
cmake --build build

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo Build successful! Executable is: build\scan.exe
endlocal
