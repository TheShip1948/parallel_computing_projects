@echo off
setlocal
echo Setting up VS environment...
call "D:\ProgramInstallation\Microsoft_build_tools_2022_17.14\VC\Auxiliary\Build\vcvars64.bat"

echo Cleaning previous build...
if exist build rmdir /s /q build
mkdir build

echo Configuring with CMake (Ninja)...
rem Since modifying CMakeLists.txt alone isn't solving the early compiler check failure 
rem in this specific environment, we pass the crucial flag here.
rem CMAKE_TryCompile_TARGET_TYPE=STATIC_LIBRARY tells CMake NOT to try to link the test program,
rem effectively skipping the step that crashes with linker errors.
cmake -G "Ninja" -S . -B build -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY

if %errorlevel% neq 0 (
    echo CMake Configuration failed!
    echo Attempting fallback to direct NVCC compilation...
    
    echo Building with NVCC...
    nvcc main.cu -o build/vector_multiply.exe
    
    if not exist build\vector_multiply.exe (
        echo Fallback Build failed!
        exit /b 1
    )
    echo Fallback Build Successful! Executable is in build/vector_multiply.exe
    exit /b 0
)

echo Building...
cmake --build build

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo Build successful! Executable is: build\vector_multiply.exe
endlocal
