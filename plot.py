import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

def plot_trajectories(filename: str = "trayectorias.dat",
                      output_png: str = "trayectorias_nbody.png",
                      output_gif: str = "simulacion_nbody.gif"):
    """
    Lee los datos espaciales y genera un gráfico estático y un GIF animado.
    """
    print(f"Procesando trayectorias desde '{filename}'...")
    try:
        # \s+ asegura que lea correctamente separaciones por uno o más espacios
        df = pd.read_csv(filename, sep=r'\s+')
        
        # Gráfico estático
        plt.figure(figsize=(8, 8))
        for particle_id, group in df.groupby('ID'):
            plt.plot(group['X'], group['Y'], label=f'Cuerpo {particle_id}')
            # Marcar inicio (x) y fin (o)
            plt.scatter(group['X'].iloc[0], group['Y'].iloc[0], marker='x', color='black')
            plt.scatter(group['X'].iloc[-1], group['Y'].iloc[-1], marker='o', s=np.clip(group['Mass'].iloc[-1]*2, 20, 200))
        
        plt.title('Trayectorias Estáticas N-Cuerpos')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.axis('equal') # Para que el espacio no se vea estirado
        
        plt.savefig(output_png, dpi=300)
        plt.close()

        # Animación GIF
        print("Generando animación GIF (esto puede tomar unos segundos)...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Determinar límites de cámara dinámicamente con un poco de margen
        margin = 5
        x_min, x_max = df['X'].min() - margin, df['X'].max() + margin
        y_min, y_max = df['Y'].min() - margin, df['Y'].max() + margin

        def update(frame):
            ax.clear()
            current_step = df[df['Step'] == frame]
            history = df[df['Step'] <= frame]
            
            # Dibujar el rastro (historial) de cada partícula
            for pid in current_step['ID']:
                p_history = history[history['ID'] == pid]
                ax.plot(p_history['X'], p_history['Y'], alpha=0.4, linewidth=1.5)
            
            # Dibujar la posición actual
            ax.scatter(current_step['X'], current_step['Y'], 
                       s=np.clip(current_step['Mass']*5, 20, 300), 
                       c=current_step['ID'], cmap='tab10', edgecolors='black', zorder=5)
            
            ax.set_title(f"Evolución N-Cuerpos (Paso {frame})")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_aspect('equal')
            return ax.collections + ax.lines

        # Filtrar frames si son muchos para que el GIF no tarde una eternidad
        unique_steps = df['Step'].unique()
        step_skip = max(1, len(unique_steps) // 150)
        frames_to_plot = unique_steps[::step_skip]

        anim = FuncAnimation(fig, update, frames=frames_to_plot, interval=50, blit=False)
        anim.save(output_gif, writer=PillowWriter(fps=20))
        plt.close()

    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe. Asegúrate de correr la simulación en C++ primero.")
    except Exception as e:
        print(f"Error al procesar '{filename}': {e}")

def plot_energy_conservation(filename: str = "energia.dat",
                             output_png: str = "energia_nbody.png"):
    """
    Lee los datos de energía y genera un gráfico comprobando la conservación.
    """
    print(f"Procesando energías desde '{filename}'...")
    try:
        df = pd.read_csv(filename, sep=r'\s+')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Step'], df['Kinetic'], label='Cinética (K)', color='blue', alpha=0.8)
        plt.plot(df['Step'], df['Potential'], label='Potencial (U)', color='orange', alpha=0.8)
        
        # La energía total debería ser una línea recta horizontal (conservación)
        plt.plot(df['Step'], df['Total'], label='Total (E = K + U)', color='black', linewidth=2, linestyle='--')
        
        plt.title('Conservación de Energía en la Simulación')
        plt.xlabel('Pasos de Simulación')
        plt.ylabel('Energía')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(output_png, dpi=300)
        plt.close()
        
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe.")
    except Exception as e:
        print(f"Error al procesar '{filename}': {e}")

def plot_performance(benchmark_file: str = "build/benchmark_results.dat",
                     scaling_file: str = "build/scaling_analysis.dat",
                     output_png: str = "performance_plots.png"):
    """
    Lee los datos de benchmarks y escalabilidad para generar gráficos de rendimiento.
    """
    print(f"Procesando rendimiento desde '{benchmark_file}' y '{scaling_file}'...")
    try:
        # Gráficos de escalabilidad
        if os.path.exists(scaling_file):
            # Parsear el bloque de escalabilidad manualmente
            threads, speedups, efficiencies = [], [], []
            with open(scaling_file, 'r') as f:
                in_scaling_block = False
                for line in f:
                    if line.startswith("# 5. Escalabilidad"):
                        in_scaling_block = True
                        continue
                    if line.startswith("# 6. Fraccion serial"):
                        in_scaling_block = False
                        break
                    
                    if in_scaling_block and not line.startswith("#") and line.strip():
                        parts = line.split()
                        if len(parts) >= 8:
                            threads.append(int(parts[0]))
                            speedups.append(float(parts[3]))
                            efficiencies.append(float(parts[5]))

            if threads:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Speedup
                ax1.plot(threads, speedups, marker='o', color='blue', label='Speedup medido')
                ax1.plot(threads, threads, linestyle='--', color='gray', label='Speedup ideal')
                ax1.set_title('Speedup vs Número de Hilos')
                ax1.set_xlabel('Hilos')
                ax1.set_ylabel('Speedup')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.6)

                # Eficiencia
                ax2.plot(threads, efficiencies, marker='s', color='green', label='Eficiencia')
                ax2.axhline(1.0, linestyle='--', color='gray', label='Eficiencia ideal')
                ax2.set_title('Eficiencia vs Número de Hilos')
                ax2.set_xlabel('Hilos')
                ax2.set_ylabel('Eficiencia')
                ax2.set_ylim([0, 1.1])
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.6)

                plt.tight_layout()
                plt.savefig(output_png, dpi=300)
                plt.close()
                print(f"Gráfico de rendimiento guardado en {output_png}")
        else:
            print(f"Error: No se encontró '{scaling_file}'")

    except Exception as e:
        print(f"Error al generar gráficos de rendimiento: {e}")

if __name__ == "__main__":
    print("--- Iniciando pipeline de visualización ---")

    # Versión base
    plot_trajectories(
        filename="trayectorias_base.dat",
        output_png="trayectorias_base.png",
        output_gif="simulacion_base.gif"
    )
    print("-" * 30)
    plot_energy_conservation(
        filename="energia_base.dat",
        output_png="energia_base.png"
    )
    print("=" * 50)

    # Versión schedule
    plot_trajectories(
        filename="trayectorias_schedule.dat",
        output_png="trayectorias_schedule.png",
        output_gif="simulacion_schedule.gif"
    )
    print("-" * 30)
    plot_energy_conservation(
        filename="energia_schedule.dat",
        output_png="energia_schedule.png"
    )
    print("=" * 50)

    # Versión chunk
    plot_trajectories(
        filename="trayectorias_chunk.dat",
        output_png="trayectorias_chunk.png",
        output_gif="simulacion_chunk.gif"
    )
    print("-" * 30)
    plot_energy_conservation(
        filename="energia_chunk.dat",
        output_png="energia_chunk.png"
    )
    print("=" * 50)

    # Versión collapse
    plot_trajectories(
        filename="trayectorias_collapse.dat",
        output_png="trayectorias_collapse.png",
        output_gif="simulacion_collapse.gif"
    )
    print("-" * 30)
    plot_energy_conservation(
        filename="energia_collapse.dat",
        output_png="energia_collapse.png"
    )
    print("=" * 50)
    #version newton3
    plot_trajectories(
        filename="trayectorias_newton3.dat",
        output_png="trayectorias_newton3.png",
        output_gif="simulacion_newton3.gif"
    )
    print("-" * 30)
    plot_energy_conservation(
        filename="energia_newton3.dat",
        output_png="energia_newton3.png"
    )
    print("=" * 50)

    # Gráficos de rendimiento
    plot_performance(
        benchmark_file="benchmark_results.dat",
        scaling_file="scaling_analysis.dat",
        output_png="performance_plots.png"
    )

    print("--- Proceso finalizado ---")