#!/usr/bin/env python3
"""
Visualizacion de benchmarks N-Body (Lab 1, OpenMP).

Genera:
  - benchmark_schedules.png
  - benchmark_synchronization.png
  - benchmark_data_clauses.png
  - benchmark_sync_advanced.png
  - benchmark_scaling.png
  - nbody_trajectories.png
  - nbody_global_state.png
  - nbody_energy_timeseries.png

El script solo grafica datos ya generados por el programa C++.
No recalcula metricas fisicas como centro de masa, radio RMS o energia.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


SECTION_RE = re.compile(r'^#\s*(\d+)\.\s*(.+?)\s*$')


def parse_dat_sections(filename):
    """Parsea un .dat con secciones '# N. Nombre' seguidas de un header
    '# col1 col2 ...' y filas de datos separadas por whitespace."""
    sections = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[!] No se encontro {filename}")
        return sections

    current_name = None
    current_header = None
    current_rows = []

    def flush():
        if current_name and current_header and current_rows:
            text = current_header + "\n" + "\n".join(current_rows)
            try:
                df = pd.read_csv(StringIO(text), sep=r'\s+', engine='python')
                sections[current_name] = df
            except Exception as e:
                print(f"[!] Error parseando '{current_name}': {e}")

    for raw in lines:
        line = raw.rstrip('\n')
        stripped = line.strip()
        if not stripped:
            continue

        m = SECTION_RE.match(stripped)
        if m:
            flush()
            current_name = m.group(2).strip()
            current_header = None
            current_rows = []
            continue

        if stripped.startswith('#'):
            if current_name is not None and current_header is None:
                current_header = stripped.lstrip('#').strip()
            continue

        if current_name is not None and current_header is not None:
            current_rows.append(stripped)

    flush()
    return sections


def read_simple_dat(filename):
    """Lee archivos .dat simples con encabezado en la primera linea."""
    if not os.path.exists(filename):
        print(f"[!] No se encontro {filename}")
        return None

    try:
        df = pd.read_csv(filename, sep=r'\s+', engine='python')
        print(f"[ok] Leido {filename}: {len(df)} filas")
        return df
    except Exception as e:
        print(f"[!] Error leyendo {filename}: {e}")
        return None


def first_existing_file(candidates):
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None


def plot_schedules(df, out='benchmark_schedules.png'):
    """Tiempo vs chunk_size para static/dynamic/guided."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for sched in ['static', 'dynamic', 'guided']:
        sub = df[df['schedule_name'] == sched].sort_values('chunk_size')
        if not sub.empty:
            ax.errorbar(
                sub['chunk_size'],
                sub['mean_s'],
                yerr=sub['stddev_s'],
                marker='o',
                capsize=4,
                label=sched
            )

    defaults = df[df['schedule_name'].str.endswith('_default')]
    for _, row in defaults.iterrows():
        ax.axhline(
            row['mean_s'],
            linestyle='--',
            alpha=0.4,
            label=f"{row['schedule_name']} (sin chunk)"
        )

    ax.set_xscale('log', base=2)
    ax.set_xlabel('chunk_size')
    ax.set_ylabel('Tiempo medio (s)')
    ax.set_title('Tiempo vs chunk_size por schedule OpenMP')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_bars(df, name_col, title, out):
    """Barras genericas con barras de error."""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = df[name_col].astype(str).values
    means = df['mean_s'].values
    errs = df['stddev_s'].values

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    bars = ax.bar(
        range(len(names)),
        means,
        yerr=errs,
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor='black'
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('Tiempo medio (s)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{m:.4f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_scaling(df_scale, df_serial=None, out='benchmark_scaling.png'):
    """Speedup, eficiencia y curva de Amdahl."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    threads = df_scale['threads'].values

    ax1.errorbar(
        threads,
        df_scale['Tp_mean'],
        yerr=df_scale['Tp_stddev'],
        marker='o',
        capsize=4,
        color='C0'
    )
    ax1.set_xlabel('Numero de threads')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempo de ejecucion vs threads')
    ax1.grid(alpha=0.3)

    ax2.errorbar(
        threads,
        df_scale['speedup'],
        yerr=df_scale['speedup_err'],
        marker='s',
        capsize=4,
        color='C1',
        label='Speedup medido'
    )
    ax2.plot(threads, threads, 'k--', alpha=0.6, label='Ideal (S=p)')

    # Extraemos f de df_serial si está disponible para graficar Gustafson
    f_val = 0.0
    if df_serial is not None and 'f_amdahl' in df_serial.columns and len(df_serial) > 0:
        f_val = df_serial['f_amdahl'].iloc[0]

    if 'amdahl_speedup' in df_scale.columns:
        ax2.plot(
            threads,
            df_scale['amdahl_speedup'],
            'r-.',
            label='Amdahl (Strong Scaling)'
        )

    if f_val > 0.0:
        # Gustafson: S_G(p) = p + (1 - p) * f
        gustafson_S = threads + (1.0 - threads) * f_val
        ax2.plot(
            threads,
            gustafson_S,
            'g:',
            linewidth=2,
            label='Gustafson (Weak Scaling)'
        )

    ax2.set_xlabel('Numero de threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs threads (Amdahl & Gustafson)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.errorbar(
        threads,
        df_scale['efficiency'] * 100,
        yerr=df_scale['eff_err'] * 100,
        marker='^',
        capsize=4,
        color='C2'
    )
    ax3.axhline(100, color='r', linestyle='--', alpha=0.5, label='100%')
    ax3.set_xlabel('Numero de threads')
    ax3.set_ylabel('Eficiencia (%)')
    ax3.set_title('Eficiencia vs threads')
    ax3.set_ylim(0, 115)
    ax3.legend()
    ax3.grid(alpha=0.3)

    if df_serial is not None and 'f_measured' in df_serial.columns:
        ax4.plot(
            df_serial['threads'],
            df_serial['f_measured'],
            marker='o',
            label='f medida'
        )
        ax4.plot(
            df_serial['threads'],
            df_serial['f_amdahl'],
            'r--',
            label='f Amdahl (global)'
        )
        ax4.set_xlabel('Numero de threads')
        ax4.set_ylabel('Fraccion serial f')
        ax4.set_title('Fraccion serial: medida vs Amdahl')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(alpha=0.3, which='both')
    else:
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_scaling_errors(df_scale, out='benchmark_scaling_errors.png'):
    """Grafica la magnitud de los errores propagados por separado."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    threads = df_scale['threads'].values

    # Grafico 1: Error de Speedup vs Threads
    ax1.plot(
        threads,
        df_scale['speedup_err'],
        marker='o',
        color='C1',
        linestyle='-',
        linewidth=2,
        label='Error de Speedup (σ_Sp)'
    )
    ax1.set_xlabel('Número de threads')
    ax1.set_ylabel('Desviación estándar del Speedup (σ_Sp)')
    ax1.set_title('Magnitud del Error del Speedup vs Threads')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Grafico 2: Error de Eficiencia vs Threads
    ax2.plot(
        threads,
        df_scale['eff_err'] * 100,  # expresado en porcentaje
        marker='^',
        color='C2',
        linestyle='-',
        linewidth=2,
        label='Error de Eficiencia (σ_Ep)'
    )
    ax2.set_xlabel('Número de threads')
    ax2.set_ylabel('Desviación estándar de Eficiencia (%)')
    ax2.set_title('Magnitud del Error de la Eficiencia vs Threads')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def select_body_ids(df_snapshots, max_bodies=12):
    """Selecciona un subconjunto de cuerpos para no saturar el grafico."""
    ids = np.sort(df_snapshots['ID'].unique())

    if len(ids) <= max_bodies:
        return ids

    idx = np.linspace(0, len(ids) - 1, max_bodies, dtype=int)
    return ids[idx]


def plot_trajectories(df_snapshots, out='nbody_trajectories.png', max_bodies=12):
    """Grafica trayectorias X-Y de un subconjunto de cuerpos."""
    required = {'Step', 'ID', 'X', 'Y', 'Mass'}
    if not required.issubset(df_snapshots.columns):
        print(f"[!] snapshots.dat no tiene las columnas esperadas: {required}")
        print(f"    columnas encontradas: {list(df_snapshots.columns)}")
        return

    selected_ids = select_body_ids(df_snapshots, max_bodies=max_bodies)

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(selected_ids)))

    for color, body_id in zip(colors, selected_ids):
        sub = df_snapshots[df_snapshots['ID'] == body_id].sort_values('Step')

        ax.plot(
            sub['X'],
            sub['Y'],
            marker='o',
            markersize=2,
            linewidth=1.2,
            alpha=0.85,
            color=color,
            label=f'ID {body_id}'
        )

        if not sub.empty:
            first = sub.iloc[0]
            last = sub.iloc[-1]

            ax.scatter(
                first['X'],
                first['Y'],
                marker='s',
                s=28,
                color=color,
                edgecolor='black',
                linewidth=0.4,
                alpha=0.9
            )

            ax.scatter(
                last['X'],
                last['Y'],
                marker='x',
                s=40,
                color=color,
                linewidth=1.2,
                alpha=0.95
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trayectorias X-Y de un subconjunto de cuerpos')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_global_state(df_energy, out='nbody_global_state.png'):
    """
    Grafica estado global usando columnas ya generadas por C++:
      CenterOfMassX, CenterOfMassY, RMSRadius, MomentumMag, MinDistance.
    """
    required = {'Step', 'CenterOfMassX', 'CenterOfMassY', 'RMSRadius'}
    if not required.issubset(df_energy.columns):
        print(f"[!] energy_timeseries.dat no tiene las columnas esperadas: {required}")
        print(f"    columnas encontradas: {list(df_energy.columns)}")
        return

    df = df_energy.sort_values('Step')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(
        df['Step'],
        df['CenterOfMassX'],
        marker='o',
        linewidth=1.5,
        markersize=3,
        label='Centro de masa X'
    )

    ax1.plot(
        df['Step'],
        df['CenterOfMassY'],
        marker='s',
        linewidth=1.5,
        markersize=3,
        label='Centro de masa Y'
    )

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Coordenada del centro de masa')
    ax1.set_title('Evolución temporal del centro de masa')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(
        df['Step'],
        df['RMSRadius'],
        marker='o',
        linewidth=1.5,
        markersize=3,
        color='C1'
    )
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Radio RMS')
    ax2.set_title('Evolucion del radio RMS')
    ax2.grid(alpha=0.3)

    if 'MomentumMag' in df.columns:
        ax3.plot(
            df['Step'],
            df['MomentumMag'],
            marker='^',
            linewidth=1.5,
            markersize=3,
            color='C2'
        )
        ax3.set_xlabel('Step')
        ax3.set_ylabel('|P|')
        ax3.set_title('Magnitud del momento lineal total')
        ax3.grid(alpha=0.3)
    else:
        ax3.axis('off')

    if 'MinDistance' in df.columns:
        ax4.plot(
            df['Step'],
            df['MinDistance'],
            marker='s',
            linewidth=1.5,
            markersize=3,
            color='C3'
        )
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Distancia minima')
        ax4.set_title('Distancia minima entre cuerpos')
        ax4.grid(alpha=0.3)
    else:
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_energy_timeseries(df_energy, out='nbody_energy_timeseries.png'):
    """
    Grafica K(t), U(t), E(t) y referencia E(0).
    No recalcula energia: solo usa columnas generadas por C++.
    """
    required = {'Step', 'Kinetic', 'Potential', 'Total'}
    if not required.issubset(df_energy.columns):
        print(f"[!] energy_timeseries.dat no tiene las columnas esperadas: {required}")
        print(f"    columnas encontradas: {list(df_energy.columns)}")
        return

    df = df_energy.sort_values('Step')

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1, ax2 = axes

    ax1.plot(
        df['Step'],
        df['Kinetic'],
        marker='o',
        markersize=3,
        linewidth=1.5,
        label='Kinetic K(t)'
    )
    ax1.plot(
        df['Step'],
        df['Potential'],
        marker='s',
        markersize=3,
        linewidth=1.5,
        label='Potential U(t)'
    )
    ax1.plot(
        df['Step'],
        df['Total'],
        marker='^',
        markersize=3,
        linewidth=1.5,
        label='Total E(t)=K+U'
    )

    ax1.axhline(
        df['Total'].iloc[0],
        color='black',
        linestyle='--',
        alpha=0.6,
        label='E(0)'
    )

    ax1.set_ylabel('Energia')
    ax1.set_title('Energia cinetica, potencial y total en el tiempo')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(
        df['Step'],
        df['Total'],
        marker='o',
        markersize=3,
        linewidth=1.5,
        color='C3',
        label='E(t)'
    )
    ax2.axhline(
        df['Total'].iloc[0],
        color='black',
        linestyle='--',
        alpha=0.6,
        label='Referencia E(0)'
    )
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Energia total')
    ax2.set_title('Deriva visual de la energia total respecto a E(0)')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_gpu_benchmarks(filename='blockdim_study.dat'):
    if not os.path.exists(filename):
        print(f"[!] No se encontro {filename}. Saltando graficos de GPU.")
        return

    try:
        df = pd.read_csv(filename, sep=r'\s+', comment='#',
                         names=['N', 'Variant', 'BlockSize', 'KernelOnlyMean', 'KernelOnlyStdDev', 'EndToEndMean', 'EndToEndStdDev', 'CpuMean', 'CpuStdDev'])
        print(f"[ok] Leido {filename}: {len(df)} filas")

        n_max = df['N'].max()
        df_n = df[df['N'] == n_max]

        plt.figure(figsize=(8, 5))
        for var, label, color in [(0, 'Basico (Global)', 'C0'), (1, 'Shared Memory', 'C1')]:
            sub = df_n[df_n['Variant'] == var].sort_values('BlockSize')
            if not sub.empty:
                plt.errorbar(sub['BlockSize'], sub['EndToEndMean'], yerr=sub['EndToEndStdDev'],
                             fmt='-o', capsize=5, label=f'{label} - End-to-End', color=color)
                plt.errorbar(sub['BlockSize'], sub['KernelOnlyMean'], yerr=sub['KernelOnlyStdDev'],
                             fmt='--s', capsize=5, label=f'{label} - Kernel Only', color=color, alpha=0.7)

        plt.xscale('log', base=2)
        plt.xticks([64, 128, 256, 512, 1024], [64, 128, 256, 512, 1024])
        plt.xlabel('blockDim.x')
        plt.ylabel('Tiempo (segundos)')
        plt.title(f'Estudio de blockDim.x vs Tiempo de Ejecucion (N={n_max})')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig('gpu_blockdim_study.png', dpi=200)
        plt.close()
        print("[ok] Generado gpu_blockdim_study.png")

        df_var1_b256 = df[(df['Variant'] == 1) & (df['BlockSize'] == 256)].sort_values('N')
        if not df_var1_b256.empty:
            plt.figure(figsize=(8, 5))
            x = np.arange(len(df_var1_b256))
            width = 0.35

            transfers = df_var1_b256['EndToEndMean'] - df_var1_b256['KernelOnlyMean']
            pct_transfers = (transfers / df_var1_b256['EndToEndMean']) * 100

            plt.bar(x - width/2, df_var1_b256['EndToEndMean'], width, label='End-to-End (con Copias)', color='#2ca02c')
            plt.bar(x + width/2, df_var1_b256['KernelOnlyMean'], width, label='Kernel-Only (Sin Copias)', color='#ff7f0e')

            for i, val in enumerate(pct_transfers):
                plt.text(i - width/2, df_var1_b256['EndToEndMean'].iloc[i] * 1.02, f"{val:.1f}% copias",
                         ha='center', fontsize=9, fontweight='bold')

            plt.xticks(x, df_var1_b256['N'])
            plt.xlabel('Tamaño de problema (N)')
            plt.ylabel('Tiempo (segundos)')
            plt.title('Impacto de Transferencias Host-Device (Shared, blockDim=256)')
            plt.legend()
            plt.grid(True, axis='y', ls='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig('gpu_transfers_impact.png', dpi=200)
            plt.close()
            print("[ok] Generado gpu_transfers_impact.png")

        df_gpu_b256 = df[df['BlockSize'] == 256]
        df_basic = df_gpu_b256[df_gpu_b256['Variant'] == 0].sort_values('N')
        df_shared = df_gpu_b256[df_gpu_b256['Variant'] == 1].sort_values('N')

        if not df_basic.empty:
            plt.figure(figsize=(8, 5))

            speedup_basic = df_basic['CpuMean'] / df_basic['EndToEndMean']
            speedup_shared = df_shared['CpuMean'] / df_shared['EndToEndMean']

            err_basic = speedup_basic * np.sqrt((df_basic['CpuStdDev']/df_basic['CpuMean'])**2 + (df_basic['EndToEndStdDev']/df_basic['EndToEndMean'])**2)
            err_shared = speedup_shared * np.sqrt((df_shared['CpuStdDev']/df_shared['CpuMean'])**2 + (df_shared['EndToEndStdDev']/df_shared['EndToEndMean'])**2)

            plt.errorbar(df_basic['N'], speedup_basic, yerr=err_basic, fmt='-o', capsize=5, label='Kernel Basico (Global)', color='C0')
            plt.errorbar(df_shared['N'], speedup_shared, yerr=err_shared, fmt='-s', capsize=5, label='Kernel Shared Memory', color='C1')

            plt.axhline(1.0, color='red', linestyle='--', alpha=0.6, label='CPU Lineal (1.0x)')
            plt.xlabel('Tamaño de problema (N)')
            plt.ylabel('Speedup (T_cpu / T_gpu_e2e)')
            plt.title('Speedup GPU vs CPU Serial (blockDim=256)')
            plt.grid(True, ls='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig('gpu_speedup_vs_n.png', dpi=200)
            plt.close()
            print("[ok] Generado gpu_speedup_vs_n.png")

        if not df_shared.empty:
            plt.figure(figsize=(8, 5))

            f = (df_shared['EndToEndMean'] - df_shared['KernelOnlyMean']) / df_shared['EndToEndMean']
            s_measured = df_shared['CpuMean'] / df_shared['EndToEndMean']
            s_kernel = df_shared['CpuMean'] / df_shared['KernelOnlyMean']

            plt.plot(df_shared['N'], s_measured, '-o', label='Speedup Medido E2E (Con overhead de copias)')
            plt.plot(df_shared['N'], s_kernel, '--s', label='Límite de Amdahl (Kernel Only, f=0)')

            plt.xlabel('Tamaño de problema (N)')
            plt.ylabel('Speedup')
            plt.title('Amdahl Limit: Impacto de la Fraccion Serial (Copias H2D/D2H)')
            plt.grid(True, ls='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig('gpu_amdahl_curve.png', dpi=200)
            plt.close()
            print("[ok] Generado gpu_amdahl_curve.png")

    except Exception as e:
        print(f"[!] Error al graficar benchmarks de GPU: {e}")


def main():
    print(">> Generando visualizaciones de benchmarks...")

    bench = parse_dat_sections('benchmark_results.dat')
    print(f"   secciones en benchmark_results.dat: {list(bench.keys())}")

    if 'Schedules' in bench:
        plot_schedules(bench['Schedules'])

    if 'Sincronizacion' in bench:
        plot_bars(
            bench['Sincronizacion'],
            'method_name',
            'Metodos de sincronizacion (atomic / critical / reduce)',
            'benchmark_synchronization.png'
        )

    if 'Clausulas de datos' in bench:
        plot_bars(
            bench['Clausulas de datos'],
            'clause_name',
            'Clausulas de datos OpenMP',
            'benchmark_data_clauses.png'
        )

    if 'Sincronizacion avanzada' in bench:
        plot_bars(
            bench['Sincronizacion avanzada'],
            'variant_name',
            'Sincronizacion avanzada (barrier/nowait/task/single)',
            'benchmark_sync_advanced.png'
        )

    scale = parse_dat_sections('scaling_analysis.dat')
    print(f"   secciones en scaling_analysis.dat: {list(scale.keys())}")

    df_scale = scale.get('Escalabilidad')
    df_serial = scale.get('Fraccion serial')

    if df_scale is not None:
        plot_scaling(df_scale, df_serial)
        plot_scaling_errors(df_scale)

    print(">> Generando visualizaciones fisicas del sistema...")

    snapshots_file = first_existing_file([
        'snapshots.dat',
        'trajectories.dat',
        'trayectorias.dat'
    ])

    energy_file = first_existing_file([
        'energy_timeseries.dat',
        'energia.dat'
    ])

    if snapshots_file is not None:
        print(f"   archivo de posiciones: {snapshots_file}")
        df_snapshots = read_simple_dat(snapshots_file)

        if df_snapshots is not None:
            plot_trajectories(df_snapshots)
    else:
        print("[!] No se encontro snapshots.dat, trajectories.dat ni trayectorias.dat")

    if energy_file is not None:
        print(f"   archivo de energia/metricas globales: {energy_file}")
        df_energy = read_simple_dat(energy_file)

        if df_energy is not None:
            plot_global_state(df_energy)
            plot_energy_timeseries(df_energy)
    else:
        print("[!] No se encontro energy_timeseries.dat ni energia.dat")

    plot_gpu_benchmarks('blockdim_study.dat')

    print(">> Listo.")


if __name__ == '__main__':
    main()