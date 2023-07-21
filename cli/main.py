import argparse
import os
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
method_folder = os.path.join(parent_folder, "method")
sys.path.append(method_folder)
import elo
import yaml
from prettytable import PrettyTable

def read_yaml(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        contenido = yaml.safe_load(archivo)
    return contenido

def run_evaluation(file):
    nombre_archivo = file.archivo_yaml
    contenido_yaml = read_yaml(nombre_archivo)
    description = contenido_yaml['example']['description']
    test_cases = contenido_yaml['example']['test_cases']
    number_of_prompts = contenido_yaml['example']['number_of_prompts']
    candidate_model = contenido_yaml['example']['candidate_model']
    generation_model = contenido_yaml['example']['generation_model']
    table = elo.generate_optimal_prompt(description, test_cases, number_of_prompts, generation_model)
    ruta_carpeta = os.path.join("..", "results")
    script_folder = os.path.dirname(os.path.abspath(__file__))
    def generar_nombre_archivo(ruta_carpeta, base_nombre="results.txt"):
        contador = 1
        while True:
            nombre_archivo = f"{base_nombre[:-4]}{contador}.txt"
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            if not os.path.exists(ruta_completa):
                return ruta_completa
            contador += 1
    ruta_completa_carpeta = os.path.join(script_folder, ruta_carpeta)
    os.makedirs(ruta_completa_carpeta, exist_ok=True)
    nombre_archivo = generar_nombre_archivo(ruta_completa_carpeta)
    tabla_str = table.get_string()
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(tabla_str) 

def main():
    parser = argparse.ArgumentParser(description="Leer archivo YAML y obtener valores de claves.")
    parser.add_argument("archivo_yaml", help="Nombre del archivo YAML a leer.")
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()