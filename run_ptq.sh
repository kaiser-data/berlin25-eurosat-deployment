#!/bin/bash

# ==========================================
# 🔬 Run Post-Training Quantization on GPU
# ==========================================
# Usage: ./run_ptq.sh <model_path>
# Example: ./run_ptq.sh outputs/fp32/2025-11-15/13-29-29/final_model.pt

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 outputs/fp32/2025-11-15/13-29-29/final_model.pt"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Submitting PTQ comparison job for: $MODEL_PATH"

# Create SLURM job
cat > ptq_job.slurm << EOF
#!/bin/bash
#SBATCH --job-name=ptq-comparison
#SBATCH --output=logs/ptq-%j.out
#SBATCH --error=logs/ptq-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_qos

echo "=========================================="
echo "PTQ Comparison Job"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Started: \$(date)"
echo "=========================================="
echo ""

# Activate environment
source ~/hackathon-venv-flwr-datasets/bin/activate

# Set MIOpen cache
export MIOPEN_USER_DB_PATH=\$SLURM_SUBMIT_DIR/.miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=\$SLURM_SUBMIT_DIR/.miopen_cache
mkdir -p \$MIOPEN_USER_DB_PATH

# Run PTQ comparison
cd \$SLURM_SUBMIT_DIR
python compare_quantizations.py $MODEL_PATH

echo ""
echo "=========================================="
echo "PTQ completed: \$(date)"
echo "=========================================="
EOF

# Create logs directory
mkdir -p logs

# Submit job
sbatch ptq_job.slurm

echo ""
echo "Job submitted! Monitor with:"
echo "  tail -f logs/ptq-*.out"
