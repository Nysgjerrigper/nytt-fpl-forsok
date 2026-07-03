@echo off
setlocal enabledelayedexpansion

echo Starting FPL Optimization with Multiple Horizon Lengths
echo =======================================================

REM Define the horizon values to test (from highest to lowest)
set horizons= 1 2 3 4 5

REM Define gameweek range - ADDED THESE TWO LINES
set start_gw= 77
set max_gw= 108

REM Create a log file for timing results
echo Horizon,Start Time,End Time,Duration > timing_results-t-auto-forced-actual.csv

REM Loop through each horizon value
for %%h in (%horizons%) do (
    echo Running with Horizon = %%h
    set start_time=!time!
    echo Start time: !start_time!
    
    REM Run the Python script WITH ADDED PARAMETERS
    python "C:\Users\peram\Documents\test\MILP Py\t-auto-forced-actual.py" --horizon %%h --start_gw !start_gw! --max_gw !max_gw!
    
    set end_time=!time!
    echo End time: !end_time!
    
    REM Calculate and display duration
    echo Horizon %%h complete! Results saved to Squad_Selection_t-auto-forced-actual%%h.csv
    echo Run time: !start_time! to !end_time!
    
    REM Log to CSV
    echo %%h,!start_time!,!end_time!,N/A >> timing_results_t-auto-forced-actual.csv
    

    echo -------------------------------------------------------
)

echo All runs completed successfully!
echo Timing results saved to timing_results-t-auto-forced-actual.csv
echo =======================================================
pause