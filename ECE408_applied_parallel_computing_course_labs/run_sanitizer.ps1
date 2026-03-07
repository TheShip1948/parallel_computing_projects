$cuda_sanitizer = "D:\ProgramInstallation\Cuda_12.5_2\bin\compute-sanitizer.bat"
$base_dir = "d:\Git\parallel_computing_projects\ECE408_applied_parallel_computing_course_labs"
$output_file = "$base_dir\sanitizer_output.txt"

# Clear output file if it exists
if (Test-Path $output_file) {
    Remove-Item $output_file
}

for ($i=1; $i -le 8; $i++) {
    $mp_dirs = Get-ChildItem -Path $base_dir -Directory -Filter "MP$i*"
    foreach ($dir in $mp_dirs) {
        Add-Content -Path $output_file -Value "=================================================" -Encoding UTF8
        Add-Content -Path $output_file -Value "Lab: $($dir.Name)" -Encoding UTF8
        Add-Content -Path $output_file -Value "=================================================" -Encoding UTF8
        
        $exes = Get-ChildItem -Path $dir.FullName -Filter "*.exe" -Recurse | Where-Object { $_.FullName -notmatch "CMakeFiles" }
        if ($exes.Count -gt 0) {
            $exe = $exes[0].FullName
            Add-Content -Path $output_file -Value "Found executable: $exe" -Encoding UTF8
            Add-Content -Path $output_file -Value "Running compute-sanitizer..." -Encoding UTF8
            
            # Execute and capture
            & $cuda_sanitizer --tool memcheck $exe 2>&1 | Out-File -FilePath $output_file -Append -Encoding UTF8
        } else {
            Add-Content -Path $output_file -Value "No executable found in $($dir.Name), skipping." -Encoding UTF8
        }
        
        Add-Content -Path $output_file -Value "" -Encoding UTF8
    }
}
