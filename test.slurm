#!/bin/bash
#SBATCH --job-name=nbody-cuda-bench
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT_DIR="$PWD"
RESULTS_DIR="$ROOT_DIR/results"

mkdir -p "$RESULTS_DIR"

# Toda la salida también se guarda como evidencia reproducible.
exec > >(tee "$RESULTS_DIR/cluster_run.log") 2>&1

echo "===== INFORMACION DE LA CORRIDA ====="
date
hostname
pwd

echo
echo "===== GPU Y CUDA ====="
nvidia-smi
nvcc --version

echo
echo "===== COMPILADORES ====="
cmake --version
c++ --version

echo
echo "===== CONFIGURACION Y COMPILACION ====="

rm -rf build

cmake -S . -B build \
  -DENABLE_CUDA=ON \
  -DBUILD_TESTING=OFF

cmake --build build \
  --parallel "${SLURM_CPUS_PER_TASK:-24}"

echo
echo "===== BENCHMARKS CUDA ====="

cd build

srun -c "${SLURM_CPUS_PER_TASK:-24}" \
  ./nbody_app --cuda-benchmarks-only

cp blockdim_study.dat \
   "$RESULTS_DIR/blockdim_study.dat"

echo
echo "===== ARCHIVOS GENERADOS ====="

ls -lh "$RESULTS_DIR"

echo
echo "===== FIN ====="

date