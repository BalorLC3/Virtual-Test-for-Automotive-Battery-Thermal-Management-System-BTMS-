# generate_driving_energy.py

import pandas as pd
import numpy as np
import os

# --- Parámetros de Conversión y Físicos Globales ---
MPH_TO_MPS = 0.44704  # 1 milla por hora = 0.44704 metros por segundo
G_ACCEL = 9.81         # Aceleración de la gravedad [m/s^2]

# ==============================================================================
# DEFAULT_VEHICLE_PARAMS: Valores base para un sedán eléctrico.
# Estos pueden ser sobrescritos o extendidos por configuraciones específicas.
# ==============================================================================
DEFAULT_VEHICLE_PARAMS = {
    'mass': 1850.0,         # Masa del vehículo [kg]
    'c_rr': 0.02,           # Coeficiente de resistencia a la rodadura
    'rho_air': 1.2,         # Densidad del aire [kg/m^3]
    'a_frontal': 2.8,       # Área frontal del vehículo [m^2]
    'c_drag': 0.35,         # Coeficiente de arrastre aerodinámico
    'drivetrain_eff': 0.80, # Eficiencia del tren motriz
    'regen_eff': 0.65       # Eficiencia del frenado regenerativo
}

# No hay directorios de entrada/salida fijos aquí, se pasarán como argumentos.

def get_power_at_wheels(v, a, params):
    """
    Calcula la potencia mecánica requerida en las ruedas (P_wheels)
    para una velocidad 'v' y aceleración 'a' dadas, usando los parámetros del vehículo.
    
    Args:
        v (float): Velocidad del vehículo [m/s].
        a (float): Aceleración del vehículo [m/s^2].
        params (dict): Diccionario con los parámetros del vehículo.
        
    Returns:
        float: Potencia requerida en las ruedas [W].
    """
    f_roll = params['mass'] * G_ACCEL * params['c_rr']
    f_aero = 0.5 * params['rho_air'] * params['a_frontal'] * params['c_drag'] * v**2
    f_accel = params['mass'] * a
    
    p_wheels = (f_roll + f_aero + f_accel) * v
    return p_wheels

def get_power_from_battery(p_wheels, params):
    """
    Calcula la potencia eléctrica requerida por la batería (P_driv)
    a partir de la potencia en las ruedas, considerando las eficiencias.
    
    Args:
        p_wheels (float): Potencia en las ruedas [W].
        params (dict): Diccionario con los parámetros del vehículo.

    Returns:
        float: Potencia requerida de la batería [W]. 
               Positiva para propulsión, negativa para regeneración.
    """
    if p_wheels >= 0:
        # Potencia de propulsión (la batería entrega energía)
        p_driv = p_wheels / params['drivetrain_eff']
    else:
        # Potencia de regeneración (la batería absorbe energía)
        p_driv = p_wheels * params['regen_eff']
    return p_driv


def process_and_save_cycles(
    input_folder, 
    vehicle_params, 
    output_dir_pot, 
    output_dir_vel, 
    target_time=2740, 
    config_suffix=""
):
    """
    Lee archivos de texto de ciclos de conducción, los procesa a perfiles de potencia
    y velocidad usando 'vehicle_params', y los guarda como archivos .npy.
    
    Args:
        input_folder (str): Ruta a la carpeta con los archivos .txt de ciclos de conducción.
        vehicle_params (dict): Diccionario con los parámetros del vehículo específicos para este procesamiento.
        output_dir_pot (str): Ruta donde se guardarán los archivos .npy de potencia.
        output_dir_vel (str): Ruta donde se guardarán los archivos .npy de velocidad.
        target_time (int): Duración deseada de los ciclos en segundos.
        config_suffix (str): Sufijo para añadir a los nombres de archivo .npy para
                             identificar la configuración del vehículo (ej. '_mass_2000kg').
                             
    Returns:
        list: Una lista de tuplas (cycle_name, power_filepath, velocity_filepath)
              para todos los ciclos procesados y guardados.
    """
    os.makedirs(output_dir_pot, exist_ok=True)
    os.makedirs(output_dir_vel, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"Advertencia: No se encontraron archivos .txt en '{input_folder}'")
        return []

    generated_files_info = []

    for file in txt_files:
        try:
            file_key = file.replace('.txt', '')
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(
                file_path, 
                sep='\s+', 
                skiprows=2, 
                names=['time_s', 'speed_mph']
            )
            # print(f"  Procesando {file_key} con sufijo '{config_suffix}'")

            # Pre-procesamiento y cálculos físicos
            df['speed_mps'] = df['speed_mph'] * MPH_TO_MPS
            dt = df['time_s'].diff().iloc[1] if df['time_s'].shape[0] > 1 else 1.0 # Handle single point case
            df['accel_mps2'] = df['speed_mps'].diff() / dt
            df.fillna({'accel_mps2':0}, inplace=True) # Rellenar el primer NaN con 0

            # Aplicar las funciones de cálculo de potencia
            df['P_wheels_W'] = df.apply(
                lambda row: get_power_at_wheels(row['speed_mps'], row['accel_mps2'], vehicle_params),
                axis=1
            )
            df['P_driv_W'] = df.apply(
                lambda row: get_power_from_battery(row['P_wheels_W'], vehicle_params),
                axis=1
            )
            
            # Extender el ciclo para alcanzar el tiempo objetivo
            p_driv_profile = df['P_driv_W'].values
            velocity_profile = df['speed_mps'].values
            
            cycle_duration = len(p_driv_profile)
            if cycle_duration == 0:
                print(f"Advertencia: Ciclo {file_key} está vacío. Saltando.")
                continue
                
            if cycle_duration < target_time:
                repeats = int(np.ceil(target_time / cycle_duration))
                p_driv_concatenated = np.tile(p_driv_profile, repeats)
                velocity_concatenated = np.tile(velocity_profile, repeats)
            else:
                p_driv_concatenated = p_driv_profile
                velocity_concatenated = velocity_profile

            # Recortar al tiempo exacto
            p_driv_final = p_driv_concatenated[:target_time]
            velocity_final = velocity_concatenated[:target_time]

            # Guardar los archivos .npy con el sufijo de configuración
            power_output_filename = f"{file_key}{config_suffix}_driving_energy.npy"
            velocity_output_filename = f"{file_key}{config_suffix}_velocity.npy"
            
            power_output_path = os.path.join(output_dir_pot, power_output_filename)
            velocity_output_path = os.path.join(output_dir_vel, velocity_output_filename)

            np.save(power_output_path, p_driv_final)
            np.save(velocity_output_path, velocity_final)
            # print(f"    Guardado: {power_output_filename}, {velocity_output_filename} ({p_driv_final.shape[0]}s)")
            
            generated_files_info.append((file_key, power_output_path, velocity_output_path))

        except Exception as e:
            print(f"Error procesando {file} para config '{config_suffix}': {e}")
            
    return generated_files_info
