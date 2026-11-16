#!/usr/bin/env bash
# ==============================================================================
# Parallel Parametric Sweep for 19Tow Rigid Mold Experiment
#
# Loops over all combinations of strain measures and contact constitutive models,
# executing up to MAX_CONCURRENT runs in parallel on GPU.
# ==============================================================================

# Concurrency & JAX Memory Settings
MAX_CONCURRENT=6
MEM_FRACTION=0.15   # ~2.4 GB cap per worker on 16GB GPU

# Parameters to sweep
STRAIN_MEASURES=("green_lagrange" "linear" "engineering" "logarithmic")
# CONTACT_MODELS=("piecewise_quadratic" "tanh" "piecewise_linear" "exponential" "constant" "linear" "quadratic")
# STRAIN_MEASURES=("green_lagrange" "linear")
CONTACT_MODELS=("piecewise_quadratic" "tanh" "piecewise_linear")

# Simulation defaults
NL_STEPS=300
PSEUDOT=60
DIR_STEP="-0.01"
DATE_STR="${DATE_STR:-$(date +'%b%-d')}"

# Formatted step strings for file names
NL_STEPS_PADDED=$(printf "%03d" "$NL_STEPS")
PSEUDOT_PADDED=$(printf "%02d" "$PSEUDOT")

# Total runs: 4 * 7 = 28 runs
TOTAL_RUNS=$((${#STRAIN_MEASURES[@]} * ${#CONTACT_MODELS[@]}))

echo "======================================================================"
echo " Starting 19Tow Parameter Sweep"
echo " Date tag: $DATE_STR"
echo " Output path: rigid_mold/$DATE_STR/<strain>/<contact>/NL${NL_STEPS_PADDED}_${PSEUDOT_PADDED}steps_CG"
echo " Total Runs: $TOTAL_RUNS ($MAX_CONCURRENT concurrent)"
echo " Memory settings: XLA_PYTHON_CLIENT_PREALLOCATE=false, MEM_FRACTION=$MEM_FRACTION"
echo "======================================================================"

mkdir -p "prof/rigid_mold/$DATE_STR"
mkdir -p "output/rigid_mold/$DATE_STR"

# Worker function for a single run
run_single() {
  local strain="$1"
  local contact="$2"
  local idx="$3"
  local total="$4"

  local log_dir="prof/rigid_mold/${DATE_STR}/${strain}/${contact}"
  mkdir -p "$log_dir"
  local log_file="${log_dir}/NL${NL_STEPS_PADDED}_${PSEUDOT_PADDED}steps_CG.log"

  echo "[$(date +'%T')] [START] ($idx/$total) Contact: $contact | Strain: $strain"

  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
  CUDA_MODULE_LOADING=LAZY \
  python experiments/sandboxes/rigidmold/19Tow.py \
    --strain-measure "$strain" \
    --contact-model "$contact" \
    --nonlinear-max-iter "$NL_STEPS" \
    --pseudoT "$PSEUDOT" \
    --dir-step "$DIR_STEP" \
    --date-str "$DATE_STR" > "$log_file" 2>&1

  local status=$?
  if [ $status -eq 0 ]; then
    echo "[$(date +'%T')] [DONE]  ($idx/$total) Contact: $contact | Strain: $strain"
  else
    echo "[$(date +'%T')] [FAIL]  ($idx/$total) Contact: $contact | Strain: $strain (exit $status, see $log_file)"
  fi
  return $status
}

export -f run_single
export NL_STEPS PSEUDOT DIR_STEP MEM_FRACTION DATE_STR NL_STEPS_PADDED PSEUDOT_PADDED

# Dispatch jobs in parallel using a bash background job queue
idx=0
for strain in "${STRAIN_MEASURES[@]}"; do
  for contact in "${CONTACT_MODELS[@]}"; do
    idx=$((idx + 1))
    
    # Run in background
    run_single "$strain" "$contact" "$idx" "$TOTAL_RUNS" &

    # Limit active concurrent background jobs
    while [ $(jobs -r -p | wc -l) -ge $MAX_CONCURRENT ]; do
      sleep 1
    done
  done
done

# Wait for remaining background jobs to finish
wait

echo ""
echo "======================================================================"
echo " Sweep Complete! Profile JSONs saved in prof/"
echo "======================================================================"
