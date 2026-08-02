#!/bin/bash
#SBATCH --job-name=nbody-cuda-bench
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="$ROOT_DIR/build_gpu"
RESULTS_DIR="$ROOT_DIR/results"
THREADS="${SLURM_CPUS_PER_TASK:-24}"

cd "$ROOT_DIR"
mkdir -p "$RESULTS_DIR"

# Guarda toda la salida como evidencia, además de slurm-%j.out.
exec > >(tee "$RESULTS_DIR/cluster_run-${SLURM_JOB_ID}.log") 2>&1

echo "===== INFORMACION DE LA CORRIDA ====="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "SLURM_CPUS_PER_TASK=$THREADS"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NO_DEFINIDO}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NO_DEFINIDO}"

echo
echo "===== GPU Y CUDA ====="
nvidia-smi -L
nvidia-smi
nvcc --version

echo
echo "===== COMPILADORES ====="
cmake --version
c++ --version

echo
echo "===== CONFIGURACION Y COMPILACION ====="
rm -rf "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80

cmake --build "$BUILD_DIR" \
  --parallel "$THREADS"

echo
echo "===== TESTS CUDA ====="
srun --ntasks=1 --cpus-per-task="$THREADS" \
  "$BUILD_DIR/nbody_gpu_tests"

echo
echo "===== SUITE COMPLETA ====="
srun --ntasks=1 --cpus-per-task="$THREADS" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure

echo
echo "===== BENCHMARKS ====="
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_DYNAMIC=false

cd "$BUILD_DIR"
srun --ntasks=1 --cpus-per-task="$THREADS" ./nbody_app

echo
echo "===== COPIA DE RESULTADOS ====="
for file in \
  blockdim_study.dat \
  benchmark_results.dat \
  scaling_analysis.dat \
  snapshots.dat \
  energy_timeseries.dat
do
  if [[ -f "$BUILD_DIR/$file" ]]; then
    cp "$BUILD_DIR/$file" "$RESULTS_DIR/$file"
  fi
done

if [[ -f "$RESULTS_DIR/blockdim_study.dat" ]]; then
  ROWS=$(grep -v '^#' "$RESULTS_DIR/blockdim_study.dat" | sed '/^[[:space:]]*$/d' | wc -l)
  echo "Combinaciones CUDA registradas: $ROWS"

  if [[ "$ROWS" -ne 40 ]]; then
    echo "ERROR: se esperaban 40 combinaciones CUDA y se obtuvieron $ROWS."
    exit 1
  fi
fi

echo
echo "===== ARCHIVOS GENERADOS ====="
ls -lh "$RESULTS_DIR"

echo
echo "===== FIN ====="
date
