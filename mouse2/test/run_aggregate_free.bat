@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo ========================================
echo Aggregate FREE trial features
echo Input : test\outputs
echo Output: test\outputs
echo ========================================

for /r "test\outputs" %%F in (*_frame_features.json) do (
    set "frame_features=%%~fF"
    set "fname=%%~nF"

    rem frame_features:
    rem user_002_XliteCrazyLight_free_B_session_01_frame_features.json
    rem task_log:
    rem user_002_XliteCrazyLight_session_01_B_task_log.json

    set "user_id="
    set "mouse_id="
    set "grip="
    set "cond="
    set "session_id="

    for /f "tokens=1,2,3,4,5,6,7,8,9 delims=_" %%a in ("!fname!") do (
        set "user_id=%%a_%%b"
        set "mouse_id=%%c"
        set "grip=%%d"
        set "cond=%%e"
        set "session_id=%%f_%%g"
    )

    if not "!grip!"=="free" (
        echo [SKIP] Not free frame features: !frame_features!
        echo.
    ) else (
        set "task_log=test\raw\!user_id!_!mouse_id!_!session_id!_!cond!_task_log.json"

        set "out=%%~fF"
        set "out=!out:_frame_features.json=_trial_features_v2.json!"

        echo Processing FREE frame features:
        echo !frame_features!
        echo User: !user_id!
        echo Mouse: !mouse_id!
        echo Condition: !cond!
        echo Task log: !task_log!
        echo Output: !out!

        if not exist "!task_log!" (
            echo [ERROR] Task log not found.
            echo Skip.
            echo.
        ) else (
            python src\aggregate_trial_features_v2.py ^
              --frame_features "!frame_features!" ^
              --task_log "!task_log!" ^
              --output "!out!"

            echo.
        )
    )
)

echo DONE aggregate FREE trial features.
pause
