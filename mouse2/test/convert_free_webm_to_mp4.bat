@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo ========================================
echo Convert all WEBM to MP4 under test\raw
echo ========================================

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ffmpeg not found. Please install ffmpeg or add it to PATH.
    pause
    exit /b 1
)

for /r "test\raw" %%F in (*_webcam.webm) do (
    set "webm=%%~fF"
    set "mp4=%%~dpnF.mp4"

    if exist "!mp4!" (
        echo [SKIP] MP4 exists: !mp4!
    ) else (
        echo Converting:
        echo !webm!
        ffmpeg -y -i "!webm!" -c:v libx264 -pix_fmt yuv420p -an "!mp4!"
        echo.
    )
)

echo DONE convert free WEBM to MP4.
pause
