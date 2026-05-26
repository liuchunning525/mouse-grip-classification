@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo ========================================
echo Extract FREE video features
echo Input : test\raw
echo Output: test\outputs
echo ========================================

for /r "test\raw" %%F in (*_webcam.mp4) do (
    set "video=%%~fF"
    set "fname=%%~nF"

    echo !fname! | findstr /i "_free_" >nul
    if errorlevel 1 (
        echo [SKIP] Not free video: !video!
        echo.
    ) else (
        rem Expected video filename:
        rem user_002_XliteCrazyLight_free_B_session_01_webcam.mp4
        rem Expected task log:
        rem user_002_XliteCrazyLight_session_01_B_task_log.json

        set "user_id="
        set "mouse_id="
        set "grip="
        set "cond="
        set "session_id="

        for /f "tokens=1,2,3,4,5,6,7,8 delims=_" %%a in ("!fname!") do (
            set "user_id=%%a_%%b"
            set "mouse_id=%%c"
            set "grip=%%d"
            set "cond=%%e"
            set "session_id=%%f_%%g"
        )

        if not "!grip!"=="free" (
            echo [SKIP] Grip is not free: !video!
            echo.
        ) else (
            if "!mouse_id!"=="G102" (
                set "ref_pose=data\outputs\ref\G102_aruco_mouse_pose_ref.json"
            ) else if "!mouse_id!"=="X2H" (
                set "ref_pose=data\outputs\ref\X2H_aruco_mouse_pose_ref.json"
            ) else if "!mouse_id!"=="XliteV3ES" (
                set "ref_pose=data\outputs\ref\XliteV3ES_aruco_mouse_pose_ref.json"
            ) else if "!mouse_id!"=="XliteCrazyLight" (
                set "ref_pose=data\outputs\ref\XliteCrazyLight_aruco_mouse_pose_ref.json"
            ) else (
                set "ref_pose="
            )

            if "!ref_pose!"=="" (
                echo [ERROR] Cannot infer mouse id:
                echo !video!
                echo Parsed mouse_id=!mouse_id!
                echo Skip.
                echo.
            ) else if not exist "!ref_pose!" (
                echo [ERROR] Ref pose not found:
                echo !ref_pose!
                echo Skip.
                echo.
            ) else (
                set "raw_dir=%%~dpF"
                set "task_log=!raw_dir!!user_id!_!mouse_id!_!session_id!_!cond!_task_log.json"

                if not exist "!task_log!" (
                    echo [ERROR] Task log not found:
                    echo !task_log!
                    echo Video:
                    echo !video!
                    echo Skip.
                    echo.
                ) else (
                    set "rel=%%~fF"
                    set "rel=!rel:%CD%\test\raw\=!"
                    set "out=test\outputs\!rel!"
                    set "out=!out:_webcam.mp4=_frame_features.json!"
                    set "preview=!out:_frame_features.json=_preview.mp4!"

                    for %%D in ("!out!") do if not exist "%%~dpD" mkdir "%%~dpD"

                    echo Processing FREE video:
                    echo !video!
                    echo User: !user_id!
                    echo Mouse: !mouse_id!
                    echo Condition: !cond!
                    echo Ref pose: !ref_pose!
                    echo Task log: !task_log!
                    echo Output: !out!

                    if exist "!out!" (
                        echo [SKIP] Frame features already exist:
                        echo !out!
                        echo.
                    ) else (
                        python src\extract_video_features_aruco.py ^
                          --video "!video!" ^
                          --ref_pose "!ref_pose!" ^
                          --hand_model "models\mediapipe\hand_landmarker.task" ^
                          --task_log "!task_log!" ^
                          --output "!out!" ^
                          --preview_video "!preview!"

                        echo.
                    )
                )
            )
        )
    )
)

echo DONE extract FREE videos.
pause