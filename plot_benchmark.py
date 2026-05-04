#!/usr/bin/env python3
"""
Visualizacion de benchmarks N-Body (Lab 1, OpenMP).
Genera los graficos requeridos por la seccion 9.1 del enunciado:
  - Tiempo vs chunk_size por schedule (static/dynamic/guided)
  - Sincronizacion: atomic / critical / reduce
  - Clausulas de datos: shared/private/firstprivate/lastprivate
  - Sincronizacion avanzada: barrier vs nowait, task vs parallel for, single
  - Speedup vs threads (con barras de error y curva Amdahl)
  - Eficiencia vs threads
  - Fraccion serial medida vs Amdahl
"""

import re
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


def plot_schedules(df, out='benchmark_schedules.png'):
    """Tiempo vs chunk_size para static/dynamic/guided (seccion 9.1.3)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for sched in ['static', 'dynamic', 'guided']:
        sub = df[df['schedule_name'] == sched].sort_values('chunk_size')
        if not sub.empty:
            ax.errorbar(sub['chunk_size'], sub['mean_s'], yerr=sub['stddev_s'],
                        marker='o', capsize=4, label=sched)

    defaults = df[df['schedule_name'].str.endswith('_default')]
    for _, row in defaults.iterrows():
        ax.axhline(row['mean_s'], linestyle='--', alpha=0.4,
                   label=f"{row['schedule_name']} (sin chunk)")

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
    bars = ax.bar(range(len(names)), means, yerr=errs, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('Tiempo medio (s)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{m:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ok] {out}")


def plot_scaling(df_scale, df_serial=None, out='benchmark_scaling.png'):
    """Speedup, eficiencia y curva de Amdahl (seccion 9.1.1, 9.1.2, 9.1.4)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    threads = df_scale['threads'].values

    ax1.errorbar(threads, df_scale['Tp_mean'], yerr=df_scale['Tp_stddev'],
                 marker='o', capsize=4, color='C0')
    ax1.set_xlabel('Numero de threads')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempo de ejecucion vs threads')
    ax1.grid(alpha=0.3)

    ax2.errorbar(threads, df_scale['speedup'], yerr=df_scale['speedup_err'],
                 marker='s', capsize=4, color='C1', label='Speedup medido')
    ax2.plot(threads, threads, 'k--', alpha=0.6, label='Ideal (S=p)')
    if 'amdahl_speedup' in df_scale.columns:
        ax2.plot(threads, df_scale['amdahl_speedup'], 'r-.',
                 label='Amdahl (f estimada)')
    ax2.set_xlabel('Numero de threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs threads (con Amdahl)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.errorbar(threads, df_scale['efficiency'] * 100,
                 yerr=df_scale['eff_err'] * 100,
                 marker='^', capsize=4, color='C2')
    ax3.axhline(100, color='r', linestyle='--', alpha=0.5, label='100%')
    ax3.set_xlabel('Numero de threads')
    ax3.set_ylabel('Eficiencia (%)')
    ax3.set_title('Eficiencia vs threads')
    ax3.set_ylim(0, 115)
    ax3.legend()
    ax3.grid(alpha=0.3)

    if df_serial is not None and 'f_measured' in df_serial.columns:
        ax4.plot(df_serial['threads'], df_serial['f_measured'],
                 marker='o', label='f medida')
        ax4.plot(df_serial['threads'], df_serial['f_amdahl'],
                 'r--', label='f Amdahl (global)')
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


def main():
    print(">> Generando visualizaciones de benchmarks...")

    bench = parse_dat_sections('benchmark_results.dat')
    print(f"   secciones en benchmark_results.dat: {list(bench.keys())}")

    if 'Schedules' in bench:
        plot_schedules(bench['Schedules'])

    if 'Sincronizacion' in bench:
        plot_bars(bench['Sincronizacion'], 'method_name',
                  'Metodos de sincronizacion (atomic / critical / reduce)',
                  'benchmark_synchronization.png')

    if 'Clausulas de datos' in bench:
        plot_bars(bench['Clausulas de datos'], 'clause_name',
                  'Clausulas de datos OpenMP',
                  'benchmark_data_clauses.png')

    if 'Sincronizacion avanzada' in bench:
        plot_bars(bench['Sincronizacion avanzada'], 'variant_name',
                  'Sincronizacion avanzada (barrier/nowait/task/single)',
                  'benchmark_sync_advanced.png')

    scale = parse_dat_sections('scaling_analysis.dat')
    print(f"   secciones en scaling_analysis.dat: {list(scale.keys())}")

    df_scale = scale.get('Escalabilidad')
    df_serial = scale.get('Fraccion serial')
    if df_scale is not None:
        plot_scaling(df_scale, df_serial)

    print(">> Listo.")


if __name__ == '__main__':
    main()
