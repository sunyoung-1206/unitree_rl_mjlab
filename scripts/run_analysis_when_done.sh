#!/bin/bash
# 학습 프로세스(PID 85763)가 끝나면 분석 자동 실행
echo "Waiting for training process (PID 85763) to finish..."
while kill -0 85763 2>/dev/null; do
    sleep 60
    echo "$(date): Training still running... $(grep -c 'Iteration time' logs/phase4_train.log) iterations logged"
done

echo "$(date): Training completed!"
echo ""
echo "Running analysis..."
MUJOCO_GL=egl python scripts/analyze_phase4.py 2>&1 | tee solver_comparison/phase4_results/analysis_log.txt

echo ""
echo "Done! Results in solver_comparison/phase4_results/"
