import argparse
import yaml
import os
from cleaner import clean_chicken_csv_files
from plotter import create_plots_for_dataset

def load_config(path):
    # Rende il percorso relativo alla cartella principale del progetto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"config\settings.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    config = load_config(args.config)

    print("Controllo pulizia dei dati...")
    success = clean_chicken_csv_files(
        raw_input_folder=config["raw_input_folder"],
        cleaned_output_folder=config["cleaned_ouput_folder"],
        artefact_threshold_seconds=config["artefact_threshold_seconds"]
    )
    
    if not success:
        print("Pulizia dei dati fallita!")
        exit(1)
    
    print("Pulizia dei dati OK!")

    print("Avvio creazione plots...")
    plot_success = create_plots_for_dataset(
        cleaned_output_folder=config["cleaned_ouput_folder"],
        time_slot_minutes=config.get("time_slot_minutes", 30),
        plot_day_window=config.get("plot_day_window", 1)  # Passa il nuovo parametro
    )
    
    if not plot_success:
        print("Creazione plots fallita!")
        exit(1)
    
    print("Operazioni completate con successo!")