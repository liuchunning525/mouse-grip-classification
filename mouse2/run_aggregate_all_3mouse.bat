@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo =========================================
echo   Aggregating ALL trials for 3 mice
echo =========================================

for /r "data\outputs" %%f in (*_frame_features.json) do (

    echo.
    echo Processing frame features:
    echo %%f

    set "out_full=%%~dpf"
    set "name=%%~nf"

    REM Mirror outputs directory back to raw directory
    set "rel_dir=!out_full:%cd%\data\outputs\=!"
    set "raw_dir=data\raw\!rel_dir!"

    set "base=!name:_frame_features=!"
    set "task_log="
    for %%j in ("!raw_dir!\*_task_log.json") do (
        set "task_log=%%~fj"
    )
    set "output=!out_full!!base!_trial_features_v2.json"

    echo Task log: !task_log!
    echo Output: !output!

    if not exist "!task_log!" (
        echo [ERROR] Task log not found.
        echo Skip.
    ) else (
        python src\aggregate_trial_features_v2.py ^
        --frame_features "%%f" ^
        --task_log "!task_log!" ^
        --output "!output!"
    )
)

echo.
echo DONE ALL AGGREGATION
pause
