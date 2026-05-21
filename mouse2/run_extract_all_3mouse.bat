@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo =========================================
echo   Extracting ALL videos for 3 mice
echo   Mouse: G102 / XliteV3ES / X2H
echo =========================================

if not exist "data\outputs" mkdir "data\outputs"

for /r "data\raw" %%f in (*_webcam.mp4) do (

    echo.
    echo Processing video:
    echo %%f

    set "raw_full=%%~dpf"
    set "name=%%~nf"

    REM Mirror raw directory to outputs directory
    set "rel_dir=!raw_full:%cd%\data\raw\=!"
    set "out_dir=data\outputs\!rel_dir!"

    if not exist "!out_dir!" mkdir "!out_dir!"

    set "task_log="
    for %%j in ("%%~dpf*_task_log.json") do (
         set "task_log=%%~fj"
    )
    set "output=!out_dir!!name:_webcam=_frame_features!.json"
    set "preview=!out_dir!!name:_webcam=_preview!.mp4"

    REM Infer mouse id from path or filename
    set "mouse_id=unknown"

    echo %%f | findstr /i "\\G102\\" >nul && set "mouse_id=G102"
    echo %%f | findstr /i "\\XliteV3ES\\" >nul && set "mouse_id=XliteV3ES"
    echo %%f | findstr /i "\\X2H\\" >nul && set "mouse_id=X2H"

    echo %%f | findstr /i "G102" >nul && set "mouse_id=G102"
    echo %%f | findstr /i "XliteV3ES" >nul && set "mouse_id=XliteV3ES"
    echo %%f | findstr /i "X2H" >nul && set "mouse_id=X2H"

    if "!mouse_id!"=="unknown" (
        echo [ERROR] Cannot infer mouse id from path or filename.
        echo Expected: G102 / XliteV3ES / X2H
        echo Skip.
    ) else (
        set "ref_pose=data\outputs\ref\!mouse_id!_aruco_mouse_pose_ref.json"

        echo Mouse: !mouse_id!
        echo Ref pose: !ref_pose!
        echo Task log: !task_log!
        echo Output: !output!

        if not exist "!ref_pose!" (
            echo [ERROR] Ref pose not found:
            echo !ref_pose!
            echo Please generate mouse reference first.
            echo Skip.
        ) else if not exist "!task_log!" (
            echo [ERROR] Task log not found:
            echo !task_log!
            echo Skip.
        ) else (
            python src\extract_video_features_aruco.py ^
            --video "%%f" ^
            --ref_pose "!ref_pose!" ^
            --hand_model "models\mediapipe\hand_landmarker.task" ^
            --task_log "!task_log!" ^
            --output "!output!" ^
            --frame_step 3 ^
            --preview_video "!preview!"
        )
    )
)

echo.
echo DONE ALL EXTRACT FOR 3 MICE
pause
