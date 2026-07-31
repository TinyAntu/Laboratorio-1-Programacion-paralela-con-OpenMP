#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <chrono>

// Constructor del benchmarking, inicializa los parámetros para ejecutar mediciones de rendimiento
Benchmark::Benchmark(int N, unsigned int seed, double G,
                     double softening, double dt, int repetitions)
    : N_bodies(N), seed(seed), G_const(G),
      softening(softening), dt(dt), repetitions(repetitions)
{
    if (repetitions < 1)
        throw std::invalid_argument("repetitions debe ser >= 1");
    if (N_bodies < 2)
        throw std::invalid_argument("N debe ser >= 2");
}

// Crea una nueva instancia del simulador de cuerpos con los parámetros configurados
NBodySimulator* Benchmark::makeSimulator() const {
    return new NBodySimulator(N_bodies, seed, G_const, softening, dt);
}

// Crea un nuevo sistema de cuerpos inicializado desde una semilla determinística
NBodySystem* Benchmark::makeSystem() const {
    NBodySystem* sys = new NBodySystem(G_const, softening);
    sys->loadFromSeed(seed, N_bodies);
    return sys;
}

// Calcula estadísticas (media y desviación estándar) a partir de un vector de tiempos medidos
TimingResult Benchmark::computeStats(const std::vector<double>& times) const {
    int n = static_cast<int>(times.size());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n;
    double sq   = 0.0;
    for (double t : times) sq += (t - mean) * (t - mean);
    double stddev = (n > 1) ? std::sqrt(sq / (n - 1)) : 0.0;
    return {mean, stddev, n};
}

// Calcula el error en el speedup usando propagación de errores
double Benchmark::speedupError(double T1, double sigT1,
                                double Tp, double sigTp) const {
    double Sp = T1 / Tp;
    return Sp * std::sqrt((sigT1 / T1) * (sigT1 / T1) +
                          (sigTp / Tp) * (sigTp / Tp));
}

// Mide el tiempo del bucle serial puro (sin paralelización) para estimar overhead de OpenMP
double Benchmark::measurePureSerialLoop() const {
    NBodySystem* sys = makeSystem();
    sys->computeAccelerations();

    double t0 = omp_get_wtime();

    sys->zeroAccelerations();
    std::vector<Particle>& bodies = sys->getBodies();
    for (auto& b : bodies) {
        b.kick(dt);
        b.drift(dt);
    }

    double elapsed = omp_get_wtime() - t0;
    delete sys;
    return elapsed;
}

// Mide el tiempo de ejecución con OpenMP configurado a un número específico de threads
double Benchmark::measureOpenMPLoop(int num_threads) const {
    NBodySystem* sys = makeSystem();

    omp_set_num_threads(num_threads);

    double t0 = omp_get_wtime();
    sys->computeAccelerations();
    double elapsed = omp_get_wtime() - t0;

    delete sys;
    return elapsed;
}

// Realiza un benchmark del tipo de schedule especificado (static/dynamic/guided)
TimingResult Benchmark::benchmarkSchedule(int schedule_type) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();
        double t0 = omp_get_wtime();
        sys->computeAccelerations(schedule_type);
        times.push_back(omp_get_wtime() - t0);
        delete sys;
    }
    return computeStats(times);
}

// Realiza un benchmark de schedule con tamaño de chunk específico
TimingResult Benchmark::benchmarkScheduleChunk(int schedule_type, int chunk_size) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();
        double t0 = omp_get_wtime();
        sys->computeAccelerations(schedule_type, chunk_size);
        times.push_back(omp_get_wtime() - t0);
        delete sys;
    }
    return computeStats(times);
}

// Benchmark completo de todos los tipos de schedule (static, dynamic, guided)
std::vector<ScheduleResult> Benchmark::benchmarkAllSchedules(
        const std::vector<int>& chunk_sizes) {

    std::vector<ScheduleResult> results;

    const std::vector<std::pair<int, std::string>> schedules = {
        {0, "static"}, {1, "dynamic"}, {2, "guided"}
    };

    for (auto& [stype, sname] : schedules) {
        results.push_back({sname + "_default", 0, benchmarkSchedule(stype)});

        for (int chunk : chunk_sizes) {
            results.push_back({sname, chunk, benchmarkScheduleChunk(stype, chunk)});
        }
    }
    return results;
}

// Realiza un benchmark de distintos métodos de sincronización
TimingResult Benchmark::benchmarkSync(int sync_type) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        double t0 = omp_get_wtime();
        sim->integrateEuler(sync_type);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de sincronización que prueba con y sin barreras explícitas
TimingResult Benchmark::benchmarkSyncBarrier(int sync_type, bool use_barrier) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        double t0 = omp_get_wtime();
        sim->integrateEuler(sync_type, use_barrier);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark del cálculo de energía del sistema con distintos métodos de reducción
TimingResult Benchmark::benchmarkEnergyMethod(int method) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->calculateEnergy(method);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de energía comparando variables privadas vs compartidas
TimingResult Benchmark::benchmarkEnergyMethodPrivate(int method, bool use_private) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->calculateEnergy(method, use_private);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark completo de todos los métodos de sincronización
std::vector<SyncResult> Benchmark::benchmarkAllSyncMethods() {
    std::vector<SyncResult> results;

    results.push_back({"integrate_atomic",    benchmarkSync(0)});
    results.push_back({"integrate_critical",  benchmarkSync(1)});
    results.push_back({"integrate_nowait",    benchmarkSync(2)});

    results.push_back({"energy_reduce",       benchmarkEnergyMethod(0)});
    results.push_back({"energy_atomic",       benchmarkEnergyMethod(1)});

    results.push_back({"energy_private_reduce",  benchmarkEnergyMethodPrivate(0, true)});
    results.push_back({"energy_private_atomic",  benchmarkEnergyMethodPrivate(1, true)});

    return results;
}

// Benchmark de acceso a datos compartidos (data sharing benchmark)
TimingResult Benchmark::benchmarkSharedAccess() {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->processBodies();
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de acceso a datos privados (private data sharing)
TimingResult Benchmark::benchmarkPrivateAccess() {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->processBodies(1);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de cláusula firstprivate
TimingResult Benchmark::benchmarkFirstprivate() {
    std::vector<double> times;
    times.reserve(repetitions);
    MetricsCalculator calc;

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        const auto& bodies = sim->getBodies();
        double G   = sim->getSystem().getG();
        double eps = sim->getSystem().getSoftening();

        double t0 = omp_get_wtime();
        calc.calculateMetricsFirstprivate(bodies, G, eps);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de cláusula lastprivate
TimingResult Benchmark::benchmarkLastprivate() {
    std::vector<double> times;
    times.reserve(repetitions);
    MetricsCalculator calc;

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        const auto& bodies = sim->getBodies();
        double G   = sim->getSystem().getG();
        double eps = sim->getSystem().getSoftening();

        double t0 = omp_get_wtime();
        calc.calculateFinalStateLastprivate(bodies, G, eps);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark completo de todas las cláusulas de datos (data sharing)
std::vector<DataClauseResult> Benchmark::benchmarkAllDataClauses() {
    return {
        {"shared",       benchmarkSharedAccess()},
        {"private",      benchmarkPrivateAccess()},
        {"firstprivate", benchmarkFirstprivate()},
        {"lastprivate",  benchmarkLastprivate()}
    };
}

// Benchmark del impacto de barreras explícitas
TimingResult Benchmark::benchmarkBarrier(bool use_barrier) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        double t0 = omp_get_wtime();
        sim->integrateEuler(0, use_barrier);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de paralelización con tasks vs parallel for
TimingResult Benchmark::benchmarkTask(int task_type) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->processBodies(task_type);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de impacto de la cláusula single en estructuras de tasks
TimingResult Benchmark::benchmarkTaskWithSingle(int task_type, bool use_single) {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        sim->integrateEuler();
        double t0 = omp_get_wtime();
        sim->processBodies(task_type, use_single);
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark de inicialización paralela con cláusula single
TimingResult Benchmark::benchmarkSingle() {
    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySimulator* sim = makeSimulator();
        double t0 = omp_get_wtime();
        sim->parallelInitializationSingle();
        times.push_back(omp_get_wtime() - t0);
        delete sim;
    }
    return computeStats(times);
}

// Benchmark completo de sincronización avanzada
std::vector<AdvSyncResult> Benchmark::benchmarkAllAdvancedSync() {
    std::vector<AdvSyncResult> results;

    results.push_back({"barrier_on",      benchmarkBarrier(true)});
    results.push_back({"nowait_off",      benchmarkBarrier(false)});
    results.push_back({"task",            benchmarkTask(0)});
    results.push_back({"parallel_for",    benchmarkTask(1)});
    results.push_back({"task_single",     benchmarkTaskWithSingle(0, true)});
    results.push_back({"task_no_single",  benchmarkTaskWithSingle(0, false)});
    results.push_back({"single_init",     benchmarkSingle()});

    {
        std::vector<double> times;
        times.reserve(repetitions);
        for (int r = 0; r < repetitions; ++r) {
            NBodySimulator* sim = makeSimulator();
            double t0 = omp_get_wtime();
            sim->simulatePhasesBarrier();
            times.push_back(omp_get_wtime() - t0);
            delete sim;
        }
        results.push_back({"simulate_phases_barrier", computeStats(times)});
    }

    return results;
}

// Benchmark de ejecución serial (sin paralelización)
TimingResult Benchmark::benchmarkSerial(int num_threads) {
    omp_set_num_threads(num_threads);

    std::vector<double> times;
    times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();
        double t0 = omp_get_wtime();
        sys->computeAccelerations();
        times.push_back(omp_get_wtime() - t0);
        delete sys;
    }

    return computeStats(times);
}

// Benchmark de ejecución paralela con OpenMP
TimingResult Benchmark::benchmarkParallel(int num_threads) {
    omp_set_num_threads(num_threads);

    std::vector<double> times;
    times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();
        double t0 = omp_get_wtime();
        sys->computeAccelerations(0);
        times.push_back(omp_get_wtime() - t0);
        delete sys;
    }

    return computeStats(times);
}

// Análisis de escalabilidad: calcula speedup y eficiencia para distintos números de threads
std::vector<ScalingResult> Benchmark::runScalingAnalysis(
        const std::vector<int>& num_threads_list) {

    std::vector<ScalingResult> results;
    results.reserve(num_threads_list.size());

    double T1 = 0.0;
    double sigT1 = 0.0;

    for (int p : num_threads_list) {
        TimingResult tr = benchmarkParallel(p);
        
        if (p == 1) {
            T1 = tr.mean;
            sigT1 = tr.stddev;
        }

        double Tp    = tr.mean;
        double sigTp = tr.stddev;

        double Sp    = (Tp > 0.0 && T1 > 0.0) ? T1 / Tp : 0.0;
        double sigSp = (Tp > 0.0 && T1 > 0.0)
                       ? speedupError(T1, sigT1, Tp, sigTp) : 0.0;
        double Ep    = (p > 0) ? Sp / p : 0.0;
        double sigEp = (p > 0) ? sigSp / p : 0.0;

        results.push_back({p, Tp, sigTp, Sp, sigSp, Ep, sigEp});
    }
    return results;
}

// Mide la fracción serial y paralela de la ejecución
SerialParallelBreakdown Benchmark::measureSerialParallelBreakdown(
        int num_threads,
        const std::vector<ScalingResult>& scaling) {

    std::vector<double> serial_times;
    serial_times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();
        sys->computeAccelerations();

        double t0 = omp_get_wtime();
        sys->zeroAccelerations();
        std::vector<Particle>& bodies = sys->getBodies();
        for (auto& b : bodies) {
            b.kick(dt);
            b.drift(dt);
        }
        serial_times.push_back(omp_get_wtime() - t0);
        delete sys;
    }
    TimingResult serial_stat = computeStats(serial_times);

    omp_set_num_threads(num_threads);

    std::vector<double> parallel_times;
    parallel_times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();

        double t0 = omp_get_wtime();
        sys->computeAccelerations();
        parallel_times.push_back(omp_get_wtime() - t0);
        delete sys;
    }
    TimingResult parallel_stat = computeStats(parallel_times);

    std::vector<double> pure_serial_times;
    std::vector<double> one_thread_times;
    pure_serial_times.reserve(repetitions);
    one_thread_times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        pure_serial_times.push_back(measurePureSerialLoop());
        one_thread_times.push_back(measureOpenMPLoop(1));
    }
    double pure_serial_mean  = computeStats(pure_serial_times).mean;
    double one_thread_mean   = computeStats(one_thread_times).mean;
    double overhead          = one_thread_mean - pure_serial_mean;
    if (overhead < 0.0) overhead = 0.0;

    double T_serial   = serial_stat.mean;
    double T_parallel = parallel_stat.mean;
    double T_total    = T_serial + T_parallel;
    double f_measured = (T_total > 0.0) ? T_serial / T_total : 0.0;

    double f_amdahl = estimateSerialFraction(scaling);

    SerialParallelBreakdown bd;
    bd.serial_time    = T_serial;
    bd.parallel_time  = T_parallel;
    bd.overhead_time  = overhead;
    bd.total_time     = T_total;
    bd.f_measured     = f_measured;
    bd.f_amdahl       = f_amdahl;
    bd.num_threads    = num_threads;
    return bd;
}

// Calcula el speedup teórico según la ley de Amdahl
double Benchmark::amdahlSpeedup(int p, double f) const {
    return 1.0 / (f + (1.0 - f) / static_cast<double>(p));
}

// Estima la fracción serial (f) del código invirtiendo la ley de Amdahl
double Benchmark::estimateSerialFraction(
        const std::vector<ScalingResult>& scaling) const {
    double sum_f = 0.0;
    int count    = 0;
    for (const auto& r : scaling) {
        if (r.num_threads <= 1 || r.speedup <= 0.0) continue;
        double inv_S = 1.0 / r.speedup;
        double inv_p = 1.0 / r.num_threads;
        if (std::abs(1.0 - inv_p) < 1e-12) continue;
        double f_i = (inv_S - inv_p) / (1.0 - inv_p);
        f_i = std::max(0.0, std::min(1.0, f_i));
        sum_f += f_i;
        ++count;
    }
    return (count > 0) ? sum_f / count : 0.0;
}

// Guarda todos los resultados de benchmark en un archivo de texto
void Benchmark::saveBenchmarkResults(
        const std::vector<ScheduleResult>&   schedule_results,
        const std::vector<SyncResult>&       sync_results,
        const std::vector<DataClauseResult>& data_results,
        const std::vector<AdvSyncResult>&    adv_results,
        const std::string& filename) const {

    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("No se pudo abrir " + filename);

    out << std::fixed << std::setprecision(6);
    out << "# Benchmark N-cuerpos 2D\n";
    out << "# N=" << N_bodies << "  seed=" << seed << "  G=" << G_const
        << "  eps=" << softening << "  dt=" << dt
        << "  rep=" << repetitions << "\n\n";

    out << "# ================================================================\n";
    out << "# 1. Schedules\n";
    out << "# schedule_name  chunk_size  mean_s  stddev_s\n";
    out << "# ================================================================\n";
    for (const auto& r : schedule_results)
        out << r.schedule_name << "\t" << r.chunk_size << "\t"
            << r.timing.mean   << "\t" << r.timing.stddev << "\n";

    out << "\n# ================================================================\n";
    out << "# 2. Sincronizacion\n";
    out << "# method_name  mean_s  stddev_s\n";
    out << "# ================================================================\n";
    for (const auto& r : sync_results)
        out << r.method_name  << "\t"
            << r.timing.mean  << "\t" << r.timing.stddev << "\n";

    out << "\n# ================================================================\n";
    out << "# 3. Clausulas de datos\n";
    out << "# clause_name  mean_s  stddev_s\n";
    out << "# ================================================================\n";
    for (const auto& r : data_results)
        out << r.clause_name  << "\t"
            << r.timing.mean  << "\t" << r.timing.stddev << "\n";

    out << "\n# ================================================================\n";
    out << "# 4. Sincronizacion avanzada\n";
    out << "# variant_name  mean_s  stddev_s\n";
    out << "# ================================================================\n";
    for (const auto& r : adv_results)
        out << r.variant_name << "\t"
            << r.timing.mean  << "\t" << r.timing.stddev << "\n";

    out.close();
    std::cout << "[Benchmark] benchmark_results guardados en " << filename << "\n";
}

// Guarda el análisis de escalabilidad en un archivo
void Benchmark::saveScalingAnalysis(
        const std::vector<ScalingResult>&           scaling,
        const std::vector<SerialParallelBreakdown>& breakdowns,
        double serial_fraction,
        const std::string& filename) const {

    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("No se pudo abrir " + filename);

    out << std::fixed << std::setprecision(6);
    out << "# Analisis de escalabilidad\n";
    out << "# N=" << N_bodies << "  seed=" << seed << "  G=" << G_const
        << "  eps=" << softening << "  dt=" << dt
        << "  rep=" << repetitions << "\n";
    out << "# Fraccion serial estimada (Amdahl): f = " << serial_fraction << "\n\n";

    out << "# ================================================================\n";
    out << "# 5. Escalabilidad\n";
    out << "# threads  Tp_mean  Tp_stddev  speedup  speedup_err  "
           "efficiency  eff_err  amdahl_speedup\n";
    out << "# ================================================================\n";
    for (const auto& r : scaling) {
        double amdahl = amdahlSpeedup(r.num_threads, serial_fraction);
        out << r.num_threads << "\t" << r.mean      << "\t" << r.stddev     << "\t"
            << r.speedup     << "\t" << r.speedup_err << "\t"
            << r.efficiency  << "\t" << r.eff_err    << "\t"
            << amdahl        << "\n";
    }

    out << "\n# ================================================================\n";
    out << "# 6. Fraccion serial\n";
    out << "# threads  serial_s  parallel_s  overhead_s  total_s  "
           "f_measured  f_amdahl\n";
    out << "# ================================================================\n";
    for (const auto& b : breakdowns) {
        out << b.num_threads   << "\t" << b.serial_time  << "\t"
            << b.parallel_time << "\t" << b.overhead_time << "\t"
            << b.total_time    << "\t" << b.f_measured    << "\t"
            << b.f_amdahl      << "\n";
    }

    out.close();
    std::cout << "[Benchmark] scaling_analysis guardado en " << filename << "\n";
}

// Ejecuta benchmarks de schedules con tamaños de chunk por defecto
std::vector<ScheduleResult> Benchmark::runScheduleBenchmarks() {
    static const std::vector<int> default_chunks = {1, 4, 8, 16, 32, 64};
    return runScheduleBenchmarks(default_chunks);
}

// Ejecuta benchmarks de schedules con tamaños de chunk personalizados
std::vector<ScheduleResult> Benchmark::runScheduleBenchmarks(
        const std::vector<int>& chunk_sizes) {
    std::cout << "[Benchmark] 1. Schedules...\n";
    return benchmarkAllSchedules(chunk_sizes);
}

// Ejecuta benchmarks de sincronización (atomic, critical, nowait, etc)
std::vector<SyncResult> Benchmark::runSyncBenchmarks() {
    std::cout << "[Benchmark] 2. Sincronizacion...\n";
    return benchmarkAllSyncMethods();
}

// Ejecuta benchmarks de cláusulas de datos (shared, private, firstprivate, lastprivate)
std::vector<DataClauseResult> Benchmark::runDataClauseBenchmarks() {
    std::cout << "[Benchmark] 3. Clausulas de datos...\n";
    return benchmarkAllDataClauses();
}

// Ejecuta benchmarks de sincronización avanzada (tasks, barriers, etc)
std::vector<AdvSyncResult> Benchmark::runAdvancedSyncBenchmarks() {
    std::cout << "[Benchmark] 4. Sincronizacion avanzada...\n";
    return benchmarkAllAdvancedSync();
}

// Ejecuta análisis completo de escalabilidad del 1 al máximo de threads
ScalingPhaseResult Benchmark::runScalingBenchmarks(int max_threads_override) {
    std::cout << "[Benchmark] 5. Escalabilidad...\n";

    int max_t = omp_get_max_threads();
    int thread_cap = max_t;
    if (max_threads_override > 0) {
        thread_cap = std::min(max_t, max_threads_override);
    }
    if (thread_cap < 1) {
        thread_cap = 1;
    }

    std::vector<int> thread_list;
    thread_list.reserve(thread_cap);
    for (int p = 1; p <= thread_cap; ++p) {
        thread_list.push_back(p);
    }

    auto scaling = runScalingAnalysis(thread_list);
    double f_amdahl = estimateSerialFraction(scaling);
    std::cout << "[Benchmark] f (Amdahl) = " << f_amdahl
              << " (threads 1.." << thread_cap << ")\n";

    std::cout << "[Benchmark] 6. Fraccion serial (instrumentacion explicita)...\n";
    std::vector<SerialParallelBreakdown> breakdowns;
    breakdowns.reserve(thread_list.size());
    for (int p : thread_list) {
        auto bd = measureSerialParallelBreakdown(p, scaling);
        breakdowns.push_back(bd);
        std::cout << "  p=" << p
                  << "  f_measured=" << bd.f_measured
                  << "  f_amdahl="   << bd.f_amdahl
                  << "  overhead="   << bd.overhead_time << " s\n";
    }

    return {scaling, breakdowns, f_amdahl};
}

// Ejecuta todos los benchmarks: schedules, sincronización, datos, sincronización avanzada y escalabilidad
void Benchmark::runAll(int max_threads_override) {
    std::cout << "[Benchmark] N=" << N_bodies
              << "  rep=" << repetitions << "\n";

    auto sched = runScheduleBenchmarks();
    auto sync  = runSyncBenchmarks();
    auto data  = runDataClauseBenchmarks();
    auto adv   = runAdvancedSyncBenchmarks();
    auto scaling_phase = runScalingBenchmarks(max_threads_override);

    saveBenchmarkResults(sched, sync, data, adv);
    saveScalingAnalysis(
        scaling_phase.scaling,
        scaling_phase.breakdowns,
        scaling_phase.serial_fraction);

    std::cout << "[Benchmark] Finalizado.\n";
}

#ifdef NBODY_HAS_CUDA
void Benchmark::warmUpGpu(int variant, int block_size) const {
    /*
     * Se lanza el kernel repetidamente durante un tiempo fijo para que la GPU
     * suba de frecuencia antes de la primera medicion. Un unico lanzamiento no
     * basta: a N pequenio dura microsegundos y la GPU sigue en reposo.
     *
     * El costo total es despreciable (una llamada por combinacion medida, no
     * por repeticion) frente a la matriz completa de benchmarks.
     */
    static constexpr double kWarmUpSeconds = 0.1;

    NBodySystem* sys = makeSystem();

    /*
     * runGpuBenchmarks captura las excepciones por combinacion y continua con
     * la siguiente, asi que un escape aqui filtraria un sistema por cada
     * combinacion omitida.
     */
    try {
        const auto t0 = std::chrono::steady_clock::now();

        for (;;) {
            (void)sys->computeAccelerationsGpuKernelOnly(
                variant,
                block_size,
                1
            );

            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - t0;

            if (elapsed.count() >= kWarmUpSeconds) {
                break;
            }
        }
    } catch (...) {
        delete sys;
        throw;
    }

    delete sys;
}
#endif

TimingResult Benchmark::benchmarkKernelOnly(
    int variant,
    int block_size,
    int steps
) {
#ifdef NBODY_HAS_CUDA
    if (steps < 1) {
        throw std::invalid_argument(
            "benchmarkKernelOnly: steps debe ser >= 1"
        );
    }

    // Rampa de reloj de la GPU: una sola vez, fuera del bucle de repeticiones.
    warmUpGpu(variant, block_size);

    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();

        /*
         * Calentamiento:
         *
         * Inicializa el contexto CUDA, reserva buffers y carga el módulo
         * antes de registrar la medición real. Basta un lanzamiento.
         */
        (void)sys->computeAccelerationsGpuKernelOnly(
            variant,
            block_size,
            1
        );

        /*
         * Medición: promedio sobre steps lanzamientos, el mismo número que
         * usa benchmarkEndToEnd. Así ambas columnas del .dat quedan en el
         * mismo régimen de muestreo y su resta mide transferencias más
         * trabajo en host en vez de ruido.
         */
        const double measured_time =
            sys->computeAccelerationsGpuKernelOnly(
                variant,
                block_size,
                steps
            );

        times.push_back(measured_time);

        delete sys;
    }

    return computeStats(times);
#else
    (void)variant;
    (void)block_size;
    (void)steps;

    throw std::runtime_error(
        "benchmarkKernelOnly: Compilado sin soporte CUDA"
    );
#endif
}

TimingResult Benchmark::benchmarkEndToEnd(
    int variant,
    int block_size,
    int steps
) {
#ifdef NBODY_HAS_CUDA
    if (steps < 1) {
        throw std::invalid_argument(
            "benchmarkEndToEnd: steps debe ser >= 1"
        );
    }

    /*
     * Mismo calentamiento que benchmarkKernelOnly: ambas mediciones deben
     * partir del mismo estado de reloj para que su resta siga aislando
     * transferencias mas trabajo en host. Tambien hace correcta esta funcion
     * cuando se la llama por separado, sin un benchmarkKernelOnly previo.
     */
    warmUpGpu(variant, block_size);

    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();

        /*
         * Calentamiento sin modificar posiciones ni velocidades del host.
         * Además deja preparados el contexto y los buffers CUDA.
         * Un solo lanzamiento basta: no es una medición.
         */
        (void)sys->computeAccelerationsGpuKernelOnly(
            variant,
            block_size,
            1
        );

        const double time_per_step =
            sys->computeAccelerationsGpuEndToEnd(
                variant,
                block_size,
                steps,
                dt
            );

        times.push_back(time_per_step);

        delete sys;
    }

    return computeStats(times);
#else
    (void)variant;
    (void)block_size;
    (void)steps;

    throw std::runtime_error(
        "benchmarkEndToEnd: Compilado sin soporte CUDA"
    );
#endif
}


TimingResult Benchmark::benchmarkCpuKernelOnly(
    int steps
) {
    if (steps < 1) {
        throw std::invalid_argument(
            "benchmarkCpuKernelOnly: steps debe ser >= 1"
        );
    }

    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();

        const auto t0 =
            std::chrono::steady_clock::now();

        /*
         * Se recalculan las aceleraciones sobre las mismas posiciones, sin
         * integrar. Es el espejo exacto de la medicion kernel-only en GPU, que
         * tambien relanza el kernel sobre un estado que no cambia.
         */
        for (int step = 0; step < steps; ++step) {
            sys->computeAccelerations();
        }

        const auto t1 =
            std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed =
            t1 - t0;

        times.push_back(
            elapsed.count() /
            static_cast<double>(steps)
        );

        delete sys;
    }

    return computeStats(times);
}


TimingResult Benchmark::benchmarkCpuEndToEnd(
    int steps
) {
    if (steps < 1) {
        throw std::invalid_argument(
            "benchmarkCpuEndToEnd: steps debe ser >= 1"
        );
    }

    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* sys = makeSystem();

        const auto t0 =
            std::chrono::steady_clock::now();

        for (int step = 0; step < steps; ++step) {
            sys->computeAccelerations();

            for (auto& body : sys->getBodies()) {
                body.kick(dt);
                body.drift(dt);
            }
        }

        const auto t1 =
            std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed =
            t1 - t0;

        times.push_back(
            elapsed.count() /
            static_cast<double>(steps)
        );

        delete sys;
    }

    return computeStats(times);
}


TimingResult Benchmark::compareCpuGpu(int n_bodies) {
#ifdef NBODY_HAS_CUDA
    // Ejecutar referencia CPU serial
    NBodySystem* cpu_sys = new NBodySystem(G_const, softening);
    cpu_sys->loadFromSeed(seed, n_bodies);
    double t0 = omp_get_wtime();
    cpu_sys->computeAccelerations();
    double cpu_time = omp_get_wtime() - t0;
    const auto& cpu_bodies = cpu_sys->getBodies();
    (void)cpu_time; // evitar warning unused

    std::vector<double> times;
    times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        NBodySystem* gpu_sys = new NBodySystem(G_const, softening);
        gpu_sys->loadFromSeed(seed, n_bodies);

        const auto gpu_t0 =
            std::chrono::steady_clock::now();

        gpu_sys->computeAccelerationsGpu(0, 256);

        const auto gpu_t1 =
            std::chrono::steady_clock::now();

        const std::chrono::duration<double> gpu_elapsed =
            gpu_t1 - gpu_t0;

        times.push_back(gpu_elapsed.count());

        if (r == 0) {
            const auto& gpu_bodies = gpu_sys->getBodies();
            for (int i = 0; i < n_bodies; ++i) {
                double diff_x = std::abs(gpu_bodies[i].getAx() - cpu_bodies[i].getAx());
                double diff_y = std::abs(gpu_bodies[i].getAy() - cpu_bodies[i].getAy());
                double tol_x = 1e-8 + 1e-4 * std::abs(cpu_bodies[i].getAx());
                double tol_y = 1e-8 + 1e-4 * std::abs(cpu_bodies[i].getAy());
                if (diff_x > tol_x || diff_y > tol_y) {
                    delete gpu_sys;
                    delete cpu_sys;
                    throw std::runtime_error("compareCpuGpu: Validacion CPU vs GPU fallo");
                }
            }
        }
        delete gpu_sys;
    }
    delete cpu_sys;
    return computeStats(times);
#else
    (void)n_bodies;
    throw std::runtime_error("compareCpuGpu: Compilado sin soporte CUDA");
#endif
}

void Benchmark::runGpuBenchmarks(
    int steps,
    const std::string& filename
) {
#ifdef NBODY_HAS_CUDA
    /*
     * El enunciado exige al menos 100 pasos para el benchmark del paso.
     */
    if (steps < 100) {
        throw std::invalid_argument(
            "runGpuBenchmarks: el benchmark final requiere steps >= 100"
        );
    }

    std::cout
        << "[Benchmark] CUDA: matriz N, variante y blockDim.x..."
        << std::endl;

    const std::vector<int> N_sizes = {
        256,
        512,
        1024,
        2000
    };

    const std::vector<int> variants = {
        0,
        1
    };

    const std::vector<int> block_sizes = {
        64,
        128,
        256,
        512,
        1024
    };

    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error(
            "No se pudo crear " + filename
        );
    }

    out << std::scientific
        << std::setprecision(10);

    out << "# Benchmarks CUDA Lab 2\n";
    out << "# variant: 0=basica, 1=shared\n";
    out << "# Los tiempos end-to-end y CPU-step "
           "son promedios por paso.\n";

    out << "# N Variant BlockSize Steps Repetitions "
           "CpuKernelMean_s CpuKernelStdDev_s "
           "CpuStepMean_s CpuStepStdDev_s "
           "KernelOnlyMean_s KernelOnlyStdDev_s "
           "EndToEndMean_s EndToEndStdDev_s "
           "KernelSpeedup KernelSpeedupErr "
           "EndToEndSpeedup EndToEndSpeedupErr "
           "SerialFraction\n";

    int valid_combinations = 0;
    int skipped_combinations = 0;

    for (int n : N_sizes) {
        std::cout
            << "  N = " << n
            << std::endl;

        /*
         * Se crea un Benchmark independiente para cada N.
         * Esto evita modificar temporalmente N_bodies.
         */
        Benchmark current(
            n,
            seed,
            G_const,
            softening,
            dt,
            repetitions
        );

        /*
         * Referencias CPU seriales del Lab 1:
         *
         * cpu_kernel: solamente aceleraciones.
         * cpu_step: aceleraciones + Euler.
         *
         * Ambas promedian el mismo numero de pasos que sus contrapartes GPU.
         * Antes cpu_kernel venia de benchmarkSerial(1), que cronometra una
         * unica llamada: a N=1024 eso lo inflaba un ~28% y producia el
         * absurdo CpuStep < CpuKernel, ademas de romper el escalado O(N^2).
         * Como cpu_kernel es el numerador de KernelSpeedup, ese sesgo se
         * propagaba a la curva de Amdahl.
         */
        const TimingResult cpu_kernel =
            current.benchmarkCpuKernelOnly(steps);

        const TimingResult cpu_step =
            current.benchmarkCpuEndToEnd(steps);

        for (int variant : variants) {
            for (int block_size : block_sizes) {
                try {
                    /*
                     * Se pasa el mismo steps que el end-to-end: ambas
                     * mediciones deben promediar la misma cantidad de
                     * lanzamientos para que SerialFraction y la curva de
                     * Amdahl tengan sentido.
                     */
                    const TimingResult kernel =
                        current.benchmarkKernelOnly(
                            variant,
                            block_size,
                            steps
                        );

                    const TimingResult end_to_end =
                        current.benchmarkEndToEnd(
                            variant,
                            block_size,
                            steps
                        );

                    const double kernel_speedup =
                        cpu_kernel.mean /
                        kernel.mean;

                    const double kernel_speedup_error =
                        current.speedupError(
                            cpu_kernel.mean,
                            cpu_kernel.stddev,
                            kernel.mean,
                            kernel.stddev
                        );

                    const double end_to_end_speedup =
                        cpu_step.mean /
                        end_to_end.mean;

                    const double end_to_end_speedup_error =
                        current.speedupError(
                            cpu_step.mean,
                            cpu_step.stddev,
                            end_to_end.mean,
                            end_to_end.stddev
                        );

                    /*
                     * Fracción no correspondiente al kernel:
                     *
                     * transferencias + sincronización + Euler host.
                     *
                     * Es una estimación de la fracción serial del paso.
                     */
                    double serial_fraction =
                        (
                            end_to_end.mean -
                            kernel.mean
                        ) /
                        end_to_end.mean;

                    serial_fraction =
                        std::max(
                            0.0,
                            std::min(
                                1.0,
                                serial_fraction
                            )
                        );

                    out
                        << n << ' '
                        << variant << ' '
                        << block_size << ' '
                        << steps << ' '
                        << repetitions << ' '

                        << cpu_kernel.mean << ' '
                        << cpu_kernel.stddev << ' '

                        << cpu_step.mean << ' '
                        << cpu_step.stddev << ' '

                        << kernel.mean << ' '
                        << kernel.stddev << ' '

                        << end_to_end.mean << ' '
                        << end_to_end.stddev << ' '

                        << kernel_speedup << ' '
                        << kernel_speedup_error << ' '

                        << end_to_end_speedup << ' '
                        << end_to_end_speedup_error << ' '

                        << serial_fraction
                        << '\n';

                    ++valid_combinations;

                    std::cout
                        << "    variant=" << variant
                        << " block=" << block_size
                        << " kernel=" << kernel.mean << " s"
                        << " e2e/step=" << end_to_end.mean << " s"
                        << std::endl;

                } catch (const std::exception& error) {
                    /*
                     * El enunciado permite omitir tamaños que la GPU
                     * rechace, siempre que quede documentado.
                     */
                    ++skipped_combinations;

                    out
                        << "# SKIPPED"
                        << " N=" << n
                        << " variant=" << variant
                        << " block=" << block_size
                        << " reason=" << error.what()
                        << '\n';

                    std::cerr
                        << "[Benchmark] Se omite"
                        << " N=" << n
                        << " variant=" << variant
                        << " block=" << block_size
                        << ": " << error.what()
                        << std::endl;
                }
            }
        }
    }

    out.close();

    std::cout
        << "[Benchmark] " << filename
        << " generado."
        << std::endl;

    std::cout
        << "[Benchmark] combinaciones validas: "
        << valid_combinations
        << std::endl;

    std::cout
        << "[Benchmark] combinaciones omitidas: "
        << skipped_combinations
        << std::endl;

#else
    (void)steps;
    (void)filename;

    std::cout
        << "[Benchmark] CUDA deshabilitado. "
           "Omitiendo benchmarks GPU."
        << std::endl;
#endif
}
