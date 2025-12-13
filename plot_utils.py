# plot_utils.py (CORRECTED)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Asumiendo que esta es tu configuración de Matplotlib
plt.rcParams.update({
    "text.usetex": False, # Setting to False for simplicity unless LaTeX is installed
    "font.family": "serif",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (10.0, 8.0), # Adjusted for better layout
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
})

def plot_results(df, dt, title, save_path=None):
    """
    Genera y guarda una gráfica con los resultados de la simulación.
    
    Args:
        df (pd.DataFrame): DataFrame con los resultados de la simulación.
        dt (float): El paso de tiempo (timestep) de la simulación en segundos.
        title (str): Título para la gráfica.
        save_path (str, optional): Ruta para guardar la imagen. Si es None, no se guarda.
    """
    if df.empty:
        print("Advertencia: El DataFrame está vacío, no se puede graficar.")
        return

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    time_axis = df['time']
    joules_to_kJ = 0.001

    # === Plot 1: Temperaturas ===
    axes[0].plot(time_axis, df['T_batt'], label='$T_{batt}$')
    axes[0].plot(time_axis, df['T_clnt'], label='$T_{clnt}$')
    axes[0].axhline(y=33.0, color='r', linestyle='--', label='$T_{des}$')
    axes[0].set_ylabel('Temperatura [$^{\circ}$C]')
    axes[0].legend()

    energy_cooling_kJ = np.cumsum(df['P_cooling']) * dt * joules_to_kJ
    axes[1].plot(time_axis, energy_cooling_kJ, label='Energía de Enfriamiento')
    axes[1].set_ylabel('Energía Acumulada [kJ]')
    axes[1].set_xlabel('Tiempo [s]')
    axes[1].legend()
    
    for ax in axes:
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    if save_path:
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()