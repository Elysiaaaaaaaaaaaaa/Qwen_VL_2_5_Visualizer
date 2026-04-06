@echo off
setlocal

REM ========== Config ==========
set "MODEL=Qwen/Qwen2.5-VL-3B-Instruct"
set "IMAGE=D:/pvzHE/test3.jpg"

REM ========== Run ==========
python check_autoprocessor_padding.py --model "%MODEL%" --image "%IMAGE%" --padding

endlocal
python .\check_autoprocessor_padding.py --model "Qwen/Qwen2.5-VL-3B-Instruct" --image "D:/pvzHE/test3.jpg"