# make_databases.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os

from generate_driving_energy import process_and_save_cycles, DEFAULT_VEHICLE_PARAMS

from sys_dynamics_casadi import BatteryThermalSystem, SystemParameters
from setup import SimConfiguration, run_simulation
from controllers import Thermostat, SMPC
from plot_utils import plot_results 

# --- CONFIGURACIÓN GENERAL ---
RAW_DRIVE_CYCLES_INPUT_PATH = r'C:\Users\super\Desktop\Supernatural\TESIS\driving_datasets' 

GENERATED_DRIVE_CYCLES_BASE_DIR = r'C:\Users\super\Desktop\Vasudeva\Xing\btms_performance\generated_drive_cycles'
GENERATED_VELOCITY_CYCLES_BASE_DIR = r'C:\Users\super\Desktop\Vasudeva\Xing\btms_performance\generated_velocity_cycles'

SIMULATION_PLOTS_DIR = r'C:\Users\super\Desktop\Vasudeva\Xing\btms_performance\simulation_plots'

os.makedirs(SIMULATION_PLOTS_DIR, exist_ok=True)
os.makedirs(GENERATED_DRIVE_CYCLES_BASE_DIR, exist_ok=True)
os.makedirs(GENERATED_VELOCITY_CYCLES_BASE_DIR, exist_ok=True)

# --- Parámetros de Simulación  ---
T_AMBIENT = 40.0
INIT_STATE = {'T_batt': 30.0, 'T_clnt': 30.0, 'soc': 0.8}
SIMULATION_DT = 1.0 # Paso de tiempo de la simulación
TARGET_CYCLE_TIME = 2740 # Duración deseada de los ciclos en segundos
T_DES = 33.0
HORIZON = 5
ALPHA_SMPC = 0.0016
N_CLUSTERS_SMPC = 4

VEHICLE_CONFIGS_TO_TEST = [
    {'name': 'Standard Sedan', 'mass': 1850.0},
    {'name': 'Heavy Sedan', 'mass': 2000.0},
    {'name': 'Light Sedan', 'mass': 1700.0},
]

def run_simulation_for_cycle_and_config(
    power_filepath, 
    velocity_filepath, 
    config_id, 
    config_name, 
    cycle_name, 
    run_id, 
    vehicle_params_summary,
    plot_results_flag=True
):
    """
    Carga datos de un ciclo de conducción específico (generado para una configuración)
    y ejecuta la simulación SMPC.
    
    Args:
        power_filepath (str): Ruta al archivo .npy del perfil de potencia.
        velocity_filepath (str): Ruta al archivo .npy del perfil de velocidad.
        config_id (int): ID numérico de la configuración del vehículo.
        config_name (str): Nombre descriptivo de la configuración del vehículo.
        cycle_name (str): Nombre base del ciclo de conducción (ej. 'UDDS').
        run_id (int): ID único para esta ejecución de simulación.
        vehicle_params_summary (str): Resumen string de los parámetros del vehículo.
        plot_results_flag (bool): Si es True, guarda un plot de los resultados.
        
    Returns:
        pd.DataFrame: DataFrame con los resultados de la simulación, incluyendo metadatos.
    """
    print(f"\n--- Ejecutando Simulación [Run ID: {run_id}] ---")
    print(f"  Config: {config_name} (ID: {config_id})")
    print(f"  Ciclo: {cycle_name}")

    try: 
        driving_data = np.load(power_filepath)
        velocity_data = np.load(velocity_filepath)
        print(f'  Datos cargados. Longitud del ciclo: {len(driving_data)}s')
    except FileNotFoundError:
        print(f"  Error: No se encontraron los archivos para la simulación. Omitiendo.")
        return None

    config = SimConfiguration(
        driving_data=driving_data,
        velocity_data=velocity_data,
        T_amb=T_AMBIENT,
        dt=SIMULATION_DT
    )

    params = SystemParameters() 

    ctrl_SMPC = SMPC(
        driving_data,
        velocity_data,
        dt=SIMULATION_DT,
        T_des=T_DES,
        horizon=HORIZON,
        alpha=ALPHA_SMPC,
        n_clusters=N_CLUSTERS_SMPC
    )

    env_smpc = BatteryThermalSystem(INIT_STATE, params)
    df_smpc = run_simulation(env_smpc, ctrl_SMPC, config)
    
    df_smpc['run_id'] = run_id
    df_smpc['vehicle_config_id'] = config_id
    df_smpc['vehicle_config_name'] = config_name
    df_smpc['driving_cycle_name'] = cycle_name
    df_smpc['vehicle_params_summary'] = vehicle_params_summary # String para referencia

    e_smpc = df_smpc['P_cooling'].sum() * SIMULATION_DT / 3.6e6 # Energía en kWh
    print(f"  Energía de enfriamiento (THERMO): {e_smpc:.4f} kWh")
    df_smpc['cooling_energy_kwh'] = e_smpc # Añadir al DataFrame para referencia

    # --- Generar plots ---
    if plot_results_flag:
        plot_title = f'SMPC - Config: {config_name}, Ciclo: {cycle_name}'
        output_filename = os.path.join(SIMULATION_PLOTS_DIR, f'smpc_run_{run_id}_{config_name.replace(" ", "_")}_{cycle_name}.png')
        plot_results(df_smpc, SIMULATION_DT, plot_title, output_filename)
        print(f"  Plot guardado en: {output_filename}")

    return df_smpc


if __name__ == "__main__":
    all_results_dfs = []
    run_id_counter = 0

    print("--- Inicio del Proceso de Generación de Base de Datos ---")
    print(f"Total de configuraciones de vehículo a probar: {len(VEHICLE_CONFIGS_TO_TEST)}")

    base_cycle_names = sorted([f.replace('.txt', '') for f in os.listdir(RAW_DRIVE_CYCLES_INPUT_PATH) if f.endswith('.txt')])
    print(f"Ciclos de conducción base encontrados: {base_cycle_names}")

    for v_idx, config_dict in enumerate(VEHICLE_CONFIGS_TO_TEST):
        config_id = v_idx + 1 # ID para la configuración (ej. 1, 2, 3...)
        config_name = config_dict.get('name', f"Custom_Config_{config_id}")

        current_vehicle_params = DEFAULT_VEHICLE_PARAMS.copy()
        current_vehicle_params.update(config_dict)

        param_summary_parts = []
        for param, value in config_dict.items():
            if param != 'name': 
                if isinstance(value, float):
                    param_summary_parts.append(f"{param}{value:.1f}".replace('.', 'p')) # ej. mass1850p0
                else:
                    param_summary_parts.append(f"{param}{value}")
        
        config_suffix = f"_vcfg{config_id}_" + "_".join(param_summary_parts) if param_summary_parts else f"_vcfg{config_id}"
        config_suffix = config_suffix.replace("__", "_") # Limpiar posibles dobles guiones
        
        vehicle_params_summary = ", ".join([f"{k}:{v}" for k, v in current_vehicle_params.items()])

        print(f"\n--- Procesando Configuración de Vehículo: '{config_name}' (ID: {config_id}) ---")
        print(f"  Parámetros: {vehicle_params_summary}")

        generated_cycles_for_config = process_and_save_cycles(
            RAW_DRIVE_CYCLES_INPUT_PATH, 
            current_vehicle_params, 
            GENERATED_DRIVE_CYCLES_BASE_DIR, 
            GENERATED_VELOCITY_CYCLES_BASE_DIR, 
            target_time=TARGET_CYCLE_TIME, 
            config_suffix=config_suffix
        )
        print(f"  {len(generated_cycles_for_config)} ciclos .npy generados para esta configuración.")

        for cycle_base_name, power_path, velocity_path in generated_cycles_for_config:
            run_id_counter += 1
            df_result = run_simulation_for_cycle_and_config(
                power_path, 
                velocity_path, 
                config_id, 
                config_name, 
                cycle_base_name, 
                run_id_counter, 
                vehicle_params_summary,
                plot_results_flag=True # Set to False if you don't want all plots
            )
            if df_result is not None:
                all_results_dfs.append(df_result)

    print("\n--- Todas las simulaciones completadas ---")

    if all_results_dfs:
        # Concatenar todos los DataFrames de resultados en uno solo
        final_database_df = pd.concat(all_results_dfs, ignore_index=True)
        
        # Guardar el DataFrame final a un archivo CSV (o a SQL)
        output_csv_path = os.path.join(SIMULATION_PLOTS_DIR, "simulation_results_database.csv")
        final_database_df.to_csv(output_csv_path, index=False)
        print(f"\nBase de datos de resultados guardada en: {output_csv_path}")
        print(f"Total de registros en la base de datos: {len(final_database_df)}")

        # Opcional: Mostrar las primeras filas de la base de datos
        print("\nPrimeras 5 filas de la base de datos de resultados:")
        print(final_database_df.head())
        print("\nColumnas disponibles:")
        print(final_database_df.columns.tolist())

        engine = create_engine('sqlite:///simulation_results.db')
        final_database_df.to_sql('smpc_simulations', con=engine, if_exists='replace', index=False)
        print("\nBase de datos de resultados también guardada en 'simulation_results.db' (SQLite).")

    else:
        print("No se generaron resultados de simulación.")

    print("\n--- Proceso finalizado ---")
    plt.show() # Muestra cualquier plot que haya quedado abierto (normalmente plot_results los cierra)