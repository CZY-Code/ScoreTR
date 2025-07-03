#!/bin/bash
RANKS=(3 5 8 10)
LEVELS=(1 2 5 10)
SIGMAS=(0.1 0.2 0.5 1.0)
GPUS=(0 1 2 3)
PIDS=()

# 捕获 SIGINT 信号并定义处理函数
trap 'handle_interrupt' INT

handle_interrupt() {
    echo "Interrupt signal received. Terminating all subprocesses..."
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    exit 1
}

# Loop through the cases and gpus arrays
# for i in "${!RANKS[@]}"; do #获取数组的所有索引值
for i in {0..2}; do
    # python exp_beijing_air.py --dev "${GPUS[$i]}" --level 2 --sigma "${SIGMAS[$i]}" &
    # python exp_click.py --dev "${GPUS[$i]}" --level 1 --sigma "${SIGMAS[$i]}" &
    # python exp_alog.py --dev "${GPUS[$i]}" --level 10 --sigma "${SIGMAS[$i]}" &
    # python exp_acc.py --dev "${GPUS[$i]}" --level "${LEVELS[$i]}" --sigma 1.0 &
    # python exp_imgC.py --dev "${GPUS[$i]}" --level "${LEVELS[$i]}" --sigma 1.0 &
    python exp_imgD.py --case 3 --dev "${GPUS[$i]}" --level "${LEVELS[$i]}" --sigma 1.0 &
    # Save process ID
    PIDS+=($!)
done

# Wait for all background tasks to complete and check for any failed tasks
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || let "FAIL+=1"
done

# Check for failed tasks
if [ "$FAIL" -gt 0 ]; then
    echo "Some tasks failed."
    exit 1
else
    echo "All tasks completed successfully."
fi