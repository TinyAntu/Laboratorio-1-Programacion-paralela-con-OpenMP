#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include "MetricsCalculator.h"
#include <string>
#include <vector>

struct TimingResult {
    double mean;
    double stddev;
    int    repetitions;
};

struct ScalingResult {
    int    num_threads;
    double mean;
    double stddev;
    double speedup;
    double speedup_err;
    double efficiency;
    double eff_err;
};

struct ScheduleResult {
    std::string  schedule_name;
    int          chunk_size;
    TimingResult timing;
};

struct SyncResult {
    std::string  method_name;
    TimingResult timing;
};

struct DataClauseResult {
    std::string  clause_name;
    TimingResult timing;
};

struct AdvSyncResult {
    std::string  variant_name;
    TimingResult timing;
};

struct SerialParallelBreakdown {
    double serial_time;
    double parallel_time;
    double overhead_time;
    double total_time;
    double f_measured;
    double f_amdahl;
    int    num_threads;
};

struct ScalingPhaseResult {
    std::vector<ScalingResult>           scaling;
    std::vector<SerialParallelBreakdown> breakdowns;
    double                               serial_fraction;
};

class Benchmark {
public:
    Benchmark(int N, unsigned int seed, double G, double softening,
              double dt, int repetitions = 10);

    TimingResult benchmarkSchedule(int schedule_type);

    TimingResult benchmarkScheduleChunk(int schedule_type, int chunk_size);

    std::vector<ScheduleResult> benchmarkAllSchedules(
            const std::vector<int>& chunk_sizes);

    TimingResult benchmarkSync(int sync_type);

    TimingResult benchmarkSyncBarrier(int sync_type, bool use_barrier);

    TimingResult benchmarkEnergyMethod(int method);

    TimingResult benchmarkEnergyMethodPrivate(int method, bool use_private);

    std::vector<SyncResult> benchmarkAllSyncMethods();

    TimingResult benchmarkSharedAccess();

    TimingResult benchmarkPrivateAccess();

    TimingResult benchmarkFirstprivate();

    TimingResult benchmarkLastprivate();

    std::vector<DataClauseResult> benchmarkAllDataClauses();

    TimingResult benchmarkBarrier(bool use_barrier);

    TimingResult benchmarkTask(int task_type);

    TimingResult benchmarkTaskWithSingle(int task_type, bool use_single);

    TimingResult benchmarkSingle();

    std::vector<AdvSyncResult> benchmarkAllAdvancedSync();

    std::vector<ScheduleResult> runScheduleBenchmarks();
    std::vector<ScheduleResult> runScheduleBenchmarks(
            const std::vector<int>& chunk_sizes);

    std::vector<SyncResult> runSyncBenchmarks();

    std::vector<DataClauseResult> runDataClauseBenchmarks();

    std::vector<AdvSyncResult> runAdvancedSyncBenchmarks();

    ScalingPhaseResult runScalingBenchmarks(int max_threads_override = 0);

    TimingResult benchmarkSerial(int num_threads = 1);

    TimingResult benchmarkParallel(int num_threads);

    //METODOS CUDA LAB2
    //tiempo del kernel (sync incluida en la medici ́on host); promedia steps
    //lanzamientos, igual que benchmarkEndToEnd, para que ambas mediciones sean
    //comparables y (e2e - kernel) aisle transferencias + trabajo en host.
    TimingResult benchmarkKernelOnly(int variant = 0, int block_size = 256,
                                     int steps = 100);
    TimingResult benchmarkEndToEnd(
        int variant = 0,
        int block_size = 256,
        int steps = 100
    );

    // Baseline CPU serial del Lab 1, solo aceleraciones. Promedia steps
    // llamadas: es el espejo de benchmarkKernelOnly, igual que
    // benchmarkCpuEndToEnd lo es de benchmarkEndToEnd. No se reutiliza
    // benchmarkSerial porque ese cronometra una unica llamada y lo usa
    // runScalingAnalysis para los benchmarks OpenMP del Lab 1.
    TimingResult benchmarkCpuKernelOnly(
        int steps = 100
    );

    TimingResult benchmarkCpuEndToEnd(
        int steps = 100
    );

    TimingResult compareCpuGpu(
        int n_bodies
    );

    void runGpuBenchmarks(
        int steps = 100,
        const std::string& filename = "blockdim_study.dat"
    );

    std::vector<ScalingResult> runScalingAnalysis(
            const std::vector<int>& num_threads_list);

    SerialParallelBreakdown measureSerialParallelBreakdown(
            int num_threads,
            const std::vector<ScalingResult>& scaling);

    double amdahlSpeedup(int p, double f) const;

    double estimateSerialFraction(
            const std::vector<ScalingResult>& scaling) const;

    void saveBenchmarkResults(
            const std::vector<ScheduleResult>&   schedule_results,
            const std::vector<SyncResult>&       sync_results,
            const std::vector<DataClauseResult>& data_results,
            const std::vector<AdvSyncResult>&    adv_results,
            const std::string& filename = "benchmark_results.dat") const;

    void saveScalingAnalysis(
            const std::vector<ScalingResult>&           scaling,
            const std::vector<SerialParallelBreakdown>& breakdowns,
            double serial_fraction,
            const std::string& filename = "scaling_analysis.dat") const;

    void runAll(int max_threads_override = 0);

    int          getN()           const { return N_bodies; }
    unsigned int getSeed()        const { return seed; }
    int          getRepetitions() const { return repetitions; }

private:
    int          N_bodies;
    unsigned int seed;
    double       G_const;
    double       softening;
    double       dt;
    int          repetitions;

    NBodySimulator* makeSimulator() const;

    NBodySystem* makeSystem() const;

    TimingResult computeStats(const std::vector<double>& times) const;

    double speedupError(double T1, double sigT1,
                        double Tp, double sigTp) const;

    double measurePureSerialLoop() const;

    double measureOpenMPLoop(int num_threads) const;

#ifdef NBODY_HAS_CUDA
    // Devuelve la GPU a su frecuencia de trabajo antes de cronometrar.
    // runGpuBenchmarks ejecuta primero los baselines CPU (segundos de trabajo
    // en host) y durante ese rato la GPU baja de reloj; la primera
    // configuracion medida de cada N pagaba la rampa de subida, con hasta 81%
    // de ruido relativo. Es calentamiento por tiempo, no por numero de
    // lanzamientos, porque la duracion de un lanzamiento depende de N.
    void warmUpGpu(int variant, int block_size) const;
#endif
};

#endif
