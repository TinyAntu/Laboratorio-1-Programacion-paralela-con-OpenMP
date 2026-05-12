#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

echo "Ejecutando Programa de Prueba en Slurm"

echo "Nodo donde se ejecuta:"
hostname

echo "Directorio actual:"
pwd

echo "Fecha inicio:"
date

echo "Partición usada:"
echo "$SLURM_JOB_PARTITION"

echo "CPUs asignadas por Slurm:"
echo "$SLURM_CPUS_PER_TASK"

echo "Configurando y compilando con CMake..."

mkdir -p build
cd build

cmake .. -DBUILD_TESTING=OFF
cmake --build . -j "${SLURM_CPUS_PER_TASK:-24}"

echo "Ejecutando programa principal..."

THREADS=${SLURM_CPUS_PER_TASK:-24}

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_DYNAMIC=false

echo "THREADS=$THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

srun -c "$THREADS" ./nbody_app

echo "Archivos generados en build:"
ls -lh

echo "Fecha término:"
date