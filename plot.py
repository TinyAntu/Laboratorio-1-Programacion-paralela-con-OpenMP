#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def read_simple_dat(filename):
    if not os.path.exists(filename): return None
    try: return pd.read_csv(filename, sep=r'\s+', engine='python', comment='#')
    except: return None

def first_existing_file(candidates):
    for f in candidates:
        if os.path.exists(f): return f
    return None

def parse_dat_sections(filename):
    if not os.path.exists(filename): return {}
    sections = {}
    current_section = None
    data_lines = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# =================='):
                continue
            if line.startswith('# ') and len(line) > 2 and line[2].isdigit() and '. ' in line:
                if current_section and data_lines:
                    try: sections[current_section] = pd.read_csv(StringIO("\n".join(data_lines)), sep=r'\s+')
                    except: pass
                current_section = line.split('. ', 1)[-1].strip()
                data_lines = []
            elif not line.startswith('#'):
                if line.strip():
                    data_lines.append(line.strip())
            elif line.startswith('# ') and len(line.split()) > 2 and current_section:
                header = line.strip('# ').strip()
                if header.split()[0] in ['threads', 'schedule_name', 'method_name', 'clause_name', 'variant_name']:
                    data_lines.append(header)
                
    if current_section and data_lines:
        try: sections[current_section] = pd.read_csv(StringIO("\n".join(data_lines)), sep=r'\s+')
        except: pass
    return sections


# ==========================================
# GRAFICOS DE OPENMP (LAB 1)
# ==========================================
def plot_schedules(df, out='benchmark_schedules.png'):
    fig, ax = plt.subplots(figsize=(9, 6))
    for sched in ['static', 'dynamic', 'guided']:
        sub = df[df['schedule_name'] == sched].sort_values('chunk_size')
        if not sub.empty:
            ax.errorbar(sub['chunk_size'], sub['mean_s'], yerr=sub['stddev_s'], marker='o', capsize=4, label=sched)
    defaults = df[df['schedule_name'].str.endswith('_default')]
    for _, row in defaults.iterrows():
        ax.axhline(row['mean_s'], linestyle='--', alpha=0.4, label=f"{row['schedule_name']} (sin chunk)")
    ax.set_xscale('log', base=2); ax.set_xlabel('chunk_size'); ax.set_ylabel('Tiempo medio (s)')
    ax.set_title('Tiempo vs chunk_size por schedule OpenMP')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")

def plot_bars(df, name_col, title, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = df[name_col].astype(str).values
    means = df['mean_s'].values
    errs = df['stddev_s'].values
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    bars = ax.bar(range(len(names)), means, yerr=errs, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('Tiempo medio (s)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{m:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")

def plot_scaling(df_scale, df_serial=None, out='benchmark_scaling.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    threads = df_scale['threads'].values
    ax1.errorbar(threads, df_scale['Tp_mean'], yerr=df_scale['Tp_stddev'], marker='o', capsize=4, color='C0')
    ax1.set_xlabel('Numero de threads'); ax1.set_ylabel('Tiempo (s)'); ax1.set_title('Tiempo de ejecucion vs threads'); ax1.grid(alpha=0.3)
    ax2.errorbar(threads, df_scale['speedup'], yerr=df_scale['speedup_err'], marker='s', capsize=4, color='C1', label='Speedup medido')
    ax2.plot(threads, threads, 'k--', alpha=0.6, label='Ideal (S=p)')
    if 'amdahl_speedup' in df_scale.columns:
        ax2.plot(threads, df_scale['amdahl_speedup'], 'r-.', label='Amdahl (f estimada)')
    ax2.set_xlabel('Numero de threads'); ax2.set_ylabel('Speedup'); ax2.set_title('Speedup vs threads (con Amdahl)')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax3.errorbar(threads, df_scale['efficiency'] * 100, yerr=df_scale['eff_err'] * 100, marker='^', capsize=4, color='C2')
    ax3.axhline(100, color='r', linestyle='--', alpha=0.5, label='100%')
    ax3.set_xlabel('Numero de threads'); ax3.set_ylabel('Eficiencia (%)'); ax3.set_title('Eficiencia vs threads')
    ax3.set_ylim(0, 115); ax3.legend(); ax3.grid(alpha=0.3)
    if df_serial is not None and 'f_measured' in df_serial.columns:
        ax4.plot(df_serial['threads'], df_serial['f_measured'], marker='o', label='f medida')
        ax4.plot(df_serial['threads'], df_serial['f_amdahl'], 'r--', label='f Amdahl (global)')
        ax4.set_xlabel('Numero de threads'); ax4.set_ylabel('Fraccion serial f'); ax4.set_title('Fraccion serial: medida vs Amdahl')
        ax4.set_yscale('log'); ax4.legend(); ax4.grid(alpha=0.3, which='both')
    else: ax4.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")


# ==========================================
# GRAFICOS FISICOS (LAB 1 & LAB 2)
# ==========================================
def select_body_ids(df, max_bodies=12):
    ids = np.sort(df['ID'].unique())
    if len(ids) <= max_bodies: return ids
    idx = np.linspace(0, len(ids) - 1, max_bodies, dtype=int)
    return ids[idx]

def plot_trajectories(df, out='5_nbody_trajectories.png', max_bodies=12):
    required = {'Step', 'ID', 'X', 'Y', 'Mass'}
    if not required.issubset(df.columns): return
    sel_ids = select_body_ids(df, max_bodies=max_bodies)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(sel_ids)))
    for color, body_id in zip(colors, sel_ids):
        sub = df[df['ID'] == body_id].sort_values('Step')
        ax.plot(sub['X'], sub['Y'], marker='o', markersize=2, linewidth=1.2, alpha=0.85, color=color, label=f'ID {body_id}')
        if not sub.empty:
            ax.scatter(sub.iloc[0]['X'], sub.iloc[0]['Y'], marker='s', s=28, color=color, edgecolor='black', alpha=0.9)
            ax.scatter(sub.iloc[-1]['X'], sub.iloc[-1]['Y'], marker='x', s=40, color=color, linewidth=1.2, alpha=0.95)
    ax.set_title('6. Trayectorias X-Y de un subconjunto de cuerpos')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.grid(alpha=0.3); ax.axis('equal'); ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")

def plot_global_state(df, out='nbody_global_state.png'):
    required = {'Step', 'CenterOfMassX', 'CenterOfMassY', 'RMSRadius'}
    if not required.issubset(df.columns): return
    df = df.sort_values('Step')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(df['Step'], df['CenterOfMassX'], marker='o', linewidth=1.5, markersize=3, label='CM X')
    ax1.plot(df['Step'], df['CenterOfMassY'], marker='s', linewidth=1.5, markersize=3, label='CM Y')
    ax1.set_title('Evolución temporal del centro de masa'); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(df['Step'], df['RMSRadius'], marker='o', linewidth=1.5, markersize=3, color='C1')
    ax2.set_title('Evolucion del radio RMS'); ax2.grid(alpha=0.3)
    if 'MomentumMag' in df.columns:
        ax3.plot(df['Step'], df['MomentumMag'], marker='^', linewidth=1.5, markersize=3, color='C2')
        ax3.set_title('Magnitud del momento lineal total'); ax3.grid(alpha=0.3)
    else: ax3.axis('off')
    if 'MinDistance' in df.columns:
        ax4.plot(df['Step'], df['MinDistance'], marker='s', linewidth=1.5, markersize=3, color='C3')
        ax4.set_title('Distancia minima entre cuerpos'); ax4.grid(alpha=0.3)
    else: ax4.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")

def plot_energy_timeseries(df, out='5_nbody_energy.png'):
    required = {'Step', 'Kinetic', 'Potential', 'Total'}
    if not required.issubset(df.columns): return
    df = df.sort_values('Step')
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1, ax2 = axes
    ax1.plot(df['Step'], df['Kinetic'], marker='o', markersize=3, label='K(t)')
    ax1.plot(df['Step'], df['Potential'], marker='s', markersize=3, label='U(t)')
    ax1.plot(df['Step'], df['Total'], marker='^', markersize=3, label='Total E(t)')
    ax1.axhline(df['Total'].iloc[0], color='black', ls='--', alpha=0.6, label='E(0)')
    ax1.set_title('5. Energia cinetica, potencial y total'); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(df['Step'], df['Total'], marker='o', markersize=3, color='C3', label='E(t)')
    ax2.axhline(df['Total'].iloc[0], color='black', ls='--', alpha=0.6, label='E(0)')
    ax2.set_title('Deriva visual de la energia total respecto a E(0)'); ax2.grid(alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")


# ==========================================
# GRAFICOS DE CUDA (LAB 2)
# ==========================================
def plot_gpu_benchmarks(filename='blockdim_study.dat'):
    if not os.path.exists(filename): return
    cols = ['N', 'Variant', 'BlockSize', 'KMean', 'KStd', 'E2EMean', 'E2EStd', 'CMean', 'CStd']
    try: df = pd.read_csv(filename, sep=r'\s+', comment='#', names=cols)
    except: return

    bs_avail = df['BlockSize'].unique()
    if len(bs_avail) == 0: return
    target_bs = 256 if 256 in bs_avail else bs_avail[0]
    n_max = df['N'].max()
    
    df_b256 = df[df['BlockSize'] == target_bs]
    df_bas = df_b256[df_b256['Variant'] == 0].sort_values('N')
    df_shr = df_b256[df_b256['Variant'] == 1].sort_values('N')

    # Grafico 1: Speedup vs N
    plt.figure(figsize=(8, 5))
    if not df_bas.empty:
        sp_bas = df_bas['CMean'] / df_bas['E2EMean']
        err_bas = sp_bas * np.sqrt((df_bas['CStd']/df_bas['CMean'])**2 + (df_bas['E2EStd']/df_bas['E2EMean'])**2)
        plt.errorbar(df_bas['N'], sp_bas, yerr=err_bas, fmt='-o', capsize=5, label='Básica (E2E)')
    if not df_shr.empty:
        sp_shr = df_shr['CMean'] / df_shr['E2EMean']
        err_shr = sp_shr * np.sqrt((df_shr['CStd']/df_shr['CMean'])**2 + (df_shr['E2EStd']/df_shr['E2EMean'])**2)
        plt.errorbar(df_shr['N'], sp_shr, yerr=err_shr, fmt='-s', capsize=5, label='Shared Memory (E2E)')
    plt.axhline(1.0, color='red', ls='--', label='CPU Serial (1.0x)')
    plt.xscale('log', base=2)
    plt.xticks(df_b256['N'].unique(), labels=[str(n) for n in df_b256['N'].unique()])
    plt.title(f'1. Speedup GPU vs. CPU frente a N (blockDim={target_bs})')
    plt.xlabel('Número de cuerpos (N)'); plt.ylabel('Speedup')
    plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig('1_gpu_speedup_vs_n.png', dpi=200); plt.close()
    print("[ok] 1_gpu_speedup_vs_n.png")

    # Grafico 2: Kernel vs End-to-End
    if not df_shr.empty:
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df_shr))
        width = 0.35
        plt.bar(x - width/2, df_shr['E2EMean'], width, label='End-to-End', color='#2ca02c')
        plt.bar(x + width/2, df_shr['KMean'], width, label='Kernel-Only', color='#ff7f0e')
        transfers = df_shr['E2EMean'] - df_shr['KMean']
        pct_transfers = (np.maximum(transfers, 0) / df_shr['E2EMean']) * 100
        for i, val in enumerate(pct_transfers):
            plt.text(i - width/2, df_shr['E2EMean'].iloc[i]*1.02, f"{val:.1f}%", ha='center', fontsize=8, fontweight='bold')
        plt.xticks(x, df_shr['N'])
        plt.title('2. Tiempo kernel-only vs. end-to-end (Impacto Transferencias)')
        plt.xlabel('Número de cuerpos (N)'); plt.ylabel('Tiempo (s)')
        plt.legend(); plt.grid(axis='y', ls='--', alpha=0.5); plt.tight_layout()
        plt.savefig('2_gpu_kernel_vs_e2e.png', dpi=200); plt.close()
        print("[ok] 2_gpu_kernel_vs_e2e.png")

    # Grafico 3: Tiempo frente a blockDim.x
    df_nmax = df[df['N'] == n_max]
    if not df_nmax.empty:
        plt.figure(figsize=(8, 5))
        for var, lbl, col in [(0,'Básica','C0'), (1,'Shared','C1')]:
            sub = df_nmax[df_nmax['Variant'] == var].sort_values('BlockSize')
            if not sub.empty:
                plt.errorbar(sub['BlockSize'], sub['E2EMean'], yerr=sub['E2EStd'], fmt='-o', capsize=5, color=col, label=f'{lbl} (E2E)')
                plt.plot(sub['BlockSize'], sub['KMean'], '--s', color=col, alpha=0.7, label=f'{lbl} (Kernel)')
        plt.xscale('log', base=2)
        plt.xticks([64, 128, 256, 512, 1024], [64, 128, 256, 512, 1024])
        plt.title(f'3. Tiempo frente a blockDim.x (N={n_max})')
        plt.xlabel('Hilos por bloque (blockDim.x)'); plt.ylabel('Tiempo de ejecución (s)')
        plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.tight_layout()
        plt.savefig('3_gpu_time_vs_blockdim.png', dpi=200); plt.close()
        print("[ok] 3_gpu_time_vs_blockdim.png")

    # Grafico 4: Curva de Amdahl
    if not df_shr.empty:
        plt.figure(figsize=(8, 5))
        transfers = np.maximum(df_shr['E2EMean'] - df_shr['KMean'], 0)
        f_measured = transfers / df_shr['CMean']
        s_measured = df_shr['CMean'] / df_shr['E2EMean']
        s_amdahl_limit = 1.0 / f_measured.replace(0, np.nan)
        s_kernel_only = df_shr['CMean'] / df_shr['KMean']

        plt.plot(df_shr['N'], s_measured, '-o', label='Speedup Medido (E2E)', color='C0')
        plt.plot(df_shr['N'], s_kernel_only, ':s', label='Speedup Kernel Puro', color='C1')
        plt.plot(df_shr['N'], s_amdahl_limit, '--^', label='Límite Amdahl (1/f)', color='C2')
        plt.xscale('log', base=2); plt.yscale('log', base=10)
        plt.xticks(df_shr['N'], labels=[str(n) for n in df_shr['N']])
        plt.title('4. Curva de Amdahl: Predicción vs Medición')
        plt.xlabel('Número de cuerpos (N)'); plt.ylabel('Speedup (Log Scale)')
        plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend(); plt.tight_layout()
        plt.savefig('4_gpu_amdahl_curve.png', dpi=200); plt.close()
        print("[ok] 4_gpu_amdahl_curve.png")

    # Grafico 6: Basica vs Shared
    if not df_bas.empty and not df_shr.empty:
        plt.figure(figsize=(8, 5))
        plt.errorbar(df_bas['N'], df_bas['E2EMean'], yerr=df_bas['E2EStd'], fmt='-o', capsize=5, label='Básica')
        plt.errorbar(df_shr['N'], df_shr['E2EMean'], yerr=df_shr['E2EStd'], fmt='-s', capsize=5, label='Shared Memory')
        plt.xscale('log', base=2); plt.yscale('log', base=10)
        plt.xticks(df_b256['N'].unique(), labels=[str(n) for n in df_b256['N'].unique()])
        plt.title('6. Comparación de Tiempos: Básica vs Shared Memory')
        plt.xlabel('Número de cuerpos (N)'); plt.ylabel('Tiempo End-to-End (s)')
        plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend(); plt.tight_layout()
        plt.savefig('6_gpu_basic_vs_shared.png', dpi=200); plt.close()
        print("[ok] 6_gpu_basic_vs_shared.png")


def main():
    print(">> Procesando Benchmarks OpenMP (Lab 1)...")
    bench_file = first_existing_file(['benchmark_results.dat', 'benchmark_results.txt'])
    if bench_file:
        bench = parse_dat_sections(bench_file)
        if 'Schedules' in bench: plot_schedules(bench['Schedules'])
        if 'Sincronizacion' in bench: plot_bars(bench['Sincronizacion'], 'method_name', 'Sincronizacion OpenMP', 'benchmark_synchronization.png')
        if 'Clausulas de datos' in bench: plot_bars(bench['Clausulas de datos'], 'clause_name', 'Clausulas OpenMP', 'benchmark_data_clauses.png')
        if 'Sincronizacion avanzada' in bench: plot_bars(bench['Sincronizacion avanzada'], 'variant_name', 'Sync Avanzada OpenMP', 'benchmark_sync_advanced.png')
        
    scale_file = first_existing_file(['scaling_analysis.dat', 'scaling_analysis.txt'])
    if scale_file:
        scale = parse_dat_sections(scale_file)
        if 'Escalabilidad' in scale:
            plot_scaling(scale['Escalabilidad'], scale.get('Fraccion serial'))

    print(">> Procesando Analisis Físico (Lab 1 & 2)...")
    snapshots_file = first_existing_file(['snapshots.dat', 'snapshots.txt', 'trajectories.dat', 'trajectories.txt'])
    if snapshots_file: plot_trajectories(read_simple_dat(snapshots_file))
    
    energy_file = first_existing_file(['energy_timeseries.dat', 'energy_timeseries.txt', 'energia.dat', 'energia.txt'])
    if energy_file: 
        df = read_simple_dat(energy_file)
        if df is not None:
            plot_global_state(df)
            plot_energy_timeseries(df)
            
    print(">> Procesando Benchmarks CUDA (Lab 2)...")
    bench_gpu = first_existing_file(['blockdim_study.dat', 'blockdim_study.txt'])
    if bench_gpu: plot_gpu_benchmarks(bench_gpu)
    
    print(">> Todo listo.")

if __name__ == '__main__':
    main()