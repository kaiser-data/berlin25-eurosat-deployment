#!/bin/bash

# ==========================================
# 🔬 Complete Comparison Workflow
# ==========================================
#
# This script runs the complete comparison:
# 1. Train FP32, FP16 (INT8 training incompatible with Flower)
# 2. Quantize the best FP32 model with real PTQ (FP32 → FP16 → INT8)
# 3. Generate comparison plots and cost analysis
#
# Usage: ./run_comparison.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔬 Federated Learning Quantization Comparison         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Step 1: Multi-Precision Training Comparison${NC}"
echo -e "  Training with different precisions (15-min time limit):"
echo -e "    • FP32:  2 rounds  (baseline) [TESTING - 5 for final]"
echo -e "    • FP16:  2 rounds  (same rounds for fair comparison) [TESTING - 5 for final]"
echo -e "    • INT8:  PTQ only (training incompatible with Flower)"
echo ""

# Submit FP32 and FP16 jobs only
echo -e "${GREEN}→ Submitting FP32 job (2 rounds)...${NC}"
JOB_FP32=$(./submit_job.sh fp32 | grep "Submitted batch job" | awk '{print $4}')
echo -e "  Job ID: $JOB_FP32"

echo -e "${GREEN}→ Submitting FP16 job (4 rounds)...${NC}"
JOB_FP16=$(./submit_job.sh fp16 | grep "Submitted batch job" | awk '{print $4}')
echo -e "  Job ID: $JOB_FP16"

echo ""
echo -e "${YELLOW}Waiting for jobs to complete...${NC}"
echo -e "  Monitor with: ${BLUE}squeue -u team11${NC}"
echo ""

# Wait for all jobs to finish with progress updates
echo -e "${BLUE}Checking job status every 30 seconds...${NC}"
while true; do
    # Check if any of the jobs are still running
    RUNNING=$(squeue -u team11 | grep -E "$JOB_FP32|$JOB_FP16" | wc -l)

    if [ "$RUNNING" -eq 0 ]; then
        echo -e "${GREEN}All jobs completed!${NC}"
        break
    fi

    echo -e "${BLUE}[$(date +%H:%M:%S)] Still running: $RUNNING/2 jobs${NC}"
    sleep 30
done

echo -e "${GREEN}✅ All training jobs completed!${NC}"
echo ""

# Find the latest FP32 model
echo -e "${YELLOW}Step 2: Post-Training Quantization (PTQ)${NC}"
echo -e "  This shows: Real compression with actual accuracy measurement"
echo ""

FP32_MODEL=$(find outputs -name "final_model.pt" -path "*/fp32/*" | sort -r | head -1)

if [ -z "$FP32_MODEL" ]; then
    echo -e "${RED}❌ No FP32 model found!${NC}"
    exit 1
fi

echo -e "${GREEN}→ Found FP32 model: $FP32_MODEL${NC}"
echo -e "${GREEN}→ Submitting PTQ comparison job (GPU)...${NC}"
echo ""

# Submit PTQ as GPU job
PTQ_JOB=$(./run_ptq.sh "$FP32_MODEL" | grep "Submitted batch job" | awk '{print $4}')
echo -e "  PTQ Job ID: $PTQ_JOB"
echo ""

# Wait for PTQ job to complete
echo -e "${BLUE}Waiting for PTQ job to complete...${NC}"
while true; do
    RUNNING=$(squeue -u team11 | grep "$PTQ_JOB" | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo -e "${GREEN}PTQ job completed!${NC}"
        break
    fi
    echo -e "${BLUE}[$(date +%H:%M:%S)] PTQ running...${NC}"
    sleep 10
done

echo ""
echo -e "${YELLOW}Step 3: Generating comparison plots${NC}"
echo ""

# Activate venv for plot generation (runs on login node, no GPU needed)
source ~/hackathon-venv-flwr-datasets/bin/activate
python create_comparison_plots.py

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ✅ Comparison Complete!                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results:${NC}"
echo -e "  📊 Training comparison: Check logs/fl-*-r100-t10m-*.out"
echo -e "  📊 PTQ comparison: Check outputs/.../quantization_comparison.json"
echo -e "  📈 Plots: Check outputs/.../comparison_plots.png"
echo ""
