import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

def select_subfolder(cleaned_output_folder):
    """Permette all'utente di selezionare una sottocartella threshold_<seconds>s."""
    # Filtra solo le cartelle che iniziano con "threshold_"
    subfolders = [f for f in os.listdir(cleaned_output_folder) 
                  if os.path.isdir(os.path.join(cleaned_output_folder, f)) 
                  and f.startswith('threshold_')]
    
    if not subfolders:
        print("Nessuna sottocartella 'threshold_<seconds>s' trovata!")
        return None
    
    print("Sottocartelle disponibili:")
    for i, folder in enumerate(subfolders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = int(input("Seleziona il numero della sottocartella: ")) - 1
            if 0 <= choice < len(subfolders):
                return os.path.join(cleaned_output_folder, subfolders[choice])
            else:
                print("Selezione non valida. Riprova.")
        except ValueError:
            print("Inserisci un numero valido.")

def load_csv_data(folder_path):
    """Carica tutti i file CSV direttamente dalla cartella threshold_<seconds>s."""
    if not os.path.exists(folder_path):
        print(f"Cartella non trovata: {folder_path}")
        return None
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("Nessun file CSV trovato nella cartella!")
        return None
    
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path, sep=';', header=None, 
                           names=['data', 'ora', 'comportamento', 'id_gallina', 'id_nido'])
            
            # Converti id_gallina da float a int
            df['id_gallina'] = df['id_gallina'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
            
            all_data.append(df)
        except Exception as e:
            print(f"Errore nel leggere {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

def format_nest_id(nest_id):
    """Converte l'ID nido da formato numerico a formato X.Y"""
    # Rimuovi .0 se presente e converti a stringa
    nest_str = str(int(float(nest_id))) if isinstance(nest_id, (int, float)) else str(nest_id)
    
    if len(nest_str) == 2:
        return f"{nest_str[0]}.{nest_str[1]}"
    elif len(nest_str) == 1:
        return f"0.{nest_str}"
    else:
        return nest_str

def create_nest_preference_plot(df, output_folder):
    """Crea un bar plot delle preferenze dei nidi."""
    # Filtra solo gli ingressi nei nidi
    entries = df[df['comportamento'] == 'IN'].copy()
    
    # Formatta gli ID dei nidi
    entries['formatted_nest_id'] = entries['id_nido'].apply(format_nest_id)
    
    # Conta le visite per ogni nido
    nest_counts = entries['formatted_nest_id'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(nest_counts.index, nest_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.title('Preferenze dei Nidi - Numero di Ingressi per Nido', fontsize=16, fontweight='bold')
    plt.xlabel('ID Nido', fontsize=12)
    plt.ylabel('Numero di Ingressi', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, 'nest_preferences.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar plot salvato in: {plot_path}")

def create_clustering_heatmap(df, output_folder):
    """Crea una cluster map interattiva con Plotly."""
    # Filtra solo gli ingressi nei nidi
    entries = df[df['comportamento'] == 'IN'].copy()
    
    # Formatta gli ID dei nidi
    entries['formatted_nest_id'] = entries['id_nido'].apply(format_nest_id)
    
    # Crea una matrice gallina x nido con il conteggio delle visite
    visit_matrix = entries.groupby(['id_gallina', 'formatted_nest_id']).size().unstack(fill_value=0)
    
    if visit_matrix.empty:
        print("Nessun dato di ingresso trovato per creare la cluster map.")
        return
    
    # Normalizza i dati per il clustering
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(visit_matrix.values)
    
    # Applica il clustering
    n_clusters = min(5, len(visit_matrix))  # Massimo 5 cluster o il numero di galline
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_data)
    else:
        cluster_labels = [0] * len(visit_matrix)
    
    # Ordina le galline per cluster
    visit_matrix_sorted = visit_matrix.copy()
    visit_matrix_sorted['cluster'] = cluster_labels
    visit_matrix_sorted = visit_matrix_sorted.sort_values('cluster')
    visit_matrix_sorted = visit_matrix_sorted.drop('cluster', axis=1)
    
    # Crea la heatmap interattiva con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=visit_matrix_sorted.values,
        x=visit_matrix_sorted.columns,
        y=[f"Gallina {idx}" for idx in visit_matrix_sorted.index],
        colorscale='Viridis',
        hovertemplate='Gallina: %{y}<br>Nido: %{x}<br>Visite: %{z}<extra></extra>',
        colorbar=dict(title="Numero di Visite")
    ))
    
    fig.update_layout(
        title='Cluster Map - Frequentazione dei Nidi per Gallina',
        xaxis_title='ID Nido',
        yaxis_title='Galline',
        width=800,
        height=600,
        font=dict(size=12)
    )
    
    plot_path = os.path.join(output_folder, 'clustering_heatmap.html')
    plot(fig, filename=plot_path, auto_open=False)
    print(f"Cluster map interattiva salvata in: {plot_path}")

def create_time_slot_flow_plot(df, output_folder, time_slot_minutes=30, plot_day_window=1):
    """Crea uno o più plot temporali dei flussi IN/OUT per ogni nido divisi in slot temporali e finestre di giorni."""
    df['datetime'] = pd.to_datetime(df['data'] + ' ' + df['ora'], format='%d/%m/%Y %H:%M:%S')
    df['date'] = df['datetime'].dt.date

    # Formatta gli ID dei nidi
    df['formatted_nest_id'] = df['id_nido'].apply(format_nest_id)
    flow_data = df[df['comportamento'].isin(['IN', 'OUT'])].copy()
    if flow_data.empty:
        print("Nessun dato IN/OUT trovato per creare il plot temporale.")
        return

    nests = sorted(flow_data['formatted_nest_id'].unique())
    if not nests:
        print("Nessun nido trovato nei dati.")
        return

    # Determina il range temporale totale
    start_time = flow_data['datetime'].min().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = flow_data['datetime'].max().replace(hour=23, minute=59, second=59, microsecond=0)

    # Crea finestre di giorni
    window_delta = timedelta(days=plot_day_window)
    current_window_start = start_time
    plot_idx = 1

    while current_window_start < end_time:
        current_window_end = min(current_window_start + window_delta, end_time + timedelta(days=1))
        window_mask = (flow_data['datetime'] >= current_window_start) & (flow_data['datetime'] < current_window_end)
        window_data = flow_data[window_mask]
        if window_data.empty:
            current_window_start = current_window_end
            plot_idx += 1
            continue

        # Crea gli slot temporali per questa finestra
        slot_duration = timedelta(minutes=time_slot_minutes)
        time_slots = []
        slot_time = current_window_start
        while slot_time < current_window_end:
            time_slots.append(slot_time)
            slot_time += slot_duration
        if len(time_slots) < 2:
            current_window_start = current_window_end
            plot_idx += 1
            continue

        # Crea subplots per ogni nido
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=len(nests),
            cols=1,
            subplot_titles=[f"Nido {nest}" for nest in nests],
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        colors = {'IN': '#2E86AB', 'OUT': '#A23B72'}

        for i, nest in enumerate(nests, 1):
            nest_data = window_data[window_data['formatted_nest_id'] == nest].copy()
            slot_counts = {'IN': [], 'OUT': []}
            slot_datetimes = []
            for j in range(len(time_slots) - 1):
                slot_start = time_slots[j]
                slot_end = time_slots[j + 1]
                slot_datetimes.append(slot_start)
                slot_data = nest_data[
                    (nest_data['datetime'] >= slot_start) &
                    (nest_data['datetime'] < slot_end)
                ]
                for behavior in ['IN', 'OUT']:
                    count = len(slot_data[slot_data['comportamento'] == behavior])
                    slot_counts[behavior].append(count)
            fig.add_trace(
                go.Scatter(
                    x=slot_datetimes,
                    y=slot_counts['IN'],
                    mode='lines+markers',
                    name='Entrate (IN)' if i == 1 else '',
                    line=dict(color=colors['IN'], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'Nido {nest}<br>Tempo: %{{x}}<br>Entrate: %{{y}}<extra></extra>',
                    showlegend=(i == 1),
                    legendgroup='IN'
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=slot_datetimes,
                    y=slot_counts['OUT'],
                    mode='lines+markers',
                    name='Uscite (OUT)' if i == 1 else '',
                    line=dict(color=colors['OUT'], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'Nido {nest}<br>Tempo: %{{x}}<br>Uscite: %{{y}}<extra></extra>',
                    showlegend=(i == 1),
                    legendgroup='OUT'
                ),
                row=i, col=1
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)

        # Layout generale
        window_label = f"{current_window_start.date()}_to_{(current_window_end - timedelta(days=1)).date()}"
        fig.update_layout(
            title=f'Flussi Temporali IN/OUT per Nido ({window_label}, slot di {time_slot_minutes} minuti)',
            height=300 * len(nests),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=10)
        )
        fig.update_xaxes(title_text="Data e Ora", row=len(nests), col=1)
        for i in range(1, len(nests) + 1):
            fig.update_yaxes(title_text="N° eventi", row=i, col=1)

        plot_path = os.path.join(output_folder, f'time_slot_flows_{window_label}.html')
        plot(fig, filename=plot_path, auto_open=False)
        print(f"Plot temporale dei flussi salvato in: {plot_path}")

        current_window_start = current_window_end
        plot_idx += 1

def create_plots_for_dataset(cleaned_output_folder, selected_subfolder=None, time_slot_minutes=30, plot_day_window=1):
    """
    Funzione principale per creare i plots. 
    Può essere chiamata da un main esterno.
    
    Args:
        cleaned_output_folder (str): Percorso alla cartella principale
        selected_subfolder (str, optional): Nome della sottocartella specifica. 
                                          Se None, prompta l'utente per la selezione.
        time_slot_minutes (int): Durata degli slot temporali in minuti
    
    Returns:
        bool: True se i plots sono stati creati con successo, False altrimenti
    """
    if not os.path.exists(cleaned_output_folder):
        print(f"Il percorso specificato non esiste: {cleaned_output_folder}")
        return False
    
    # Seleziona la sottocartella
    if selected_subfolder:
        # Usa la sottocartella specificata
        folder_path = os.path.join(cleaned_output_folder, selected_subfolder)
        if not os.path.exists(folder_path):
            print(f"Sottocartella specificata non trovata: {folder_path}")
            return False
        selected_folder = folder_path
    else:
        # Prompta l'utente per la selezione
        selected_folder = select_subfolder(cleaned_output_folder)
        if not selected_folder:
            return False
    
    print(f"Cartella selezionata: {selected_folder}")
    
    # Carica i dati CSV direttamente dalla cartella threshold_<seconds>s
    df = load_csv_data(selected_folder)
    if df is None:
        return False
    
    print(f"Caricati {len(df)} record dai file CSV")
    
    # Crea/verifica la cartella plots dentro la sottocartella threshold_<seconds>s
    plots_folder = os.path.join(selected_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    print(f"Cartella plots verificata/creata in: {plots_folder}")
    
    try:
        # Crea i grafici
        print("Creazione del bar plot delle preferenze dei nidi...")
        create_nest_preference_plot(df, plots_folder)
        
        print("Creazione della cluster map interattiva...")
        create_clustering_heatmap(df, plots_folder)
        
        print(f"Creazione del plot temporale dei flussi (slot di {time_slot_minutes} minuti, finestra di {plot_day_window} giorni)...")
        create_time_slot_flow_plot(df, plots_folder, time_slot_minutes, plot_day_window)
        
        print("Tutti i grafici sono stati creati con successo!")
        return True
        
    except Exception as e:
        print(f"Errore durante la creazione dei plots: {e}")
        return False

def main():
    """Funzione principale dello script per uso standalone."""
    # Chiedi all'utente di inserire il percorso della cartella cleaned_output_folder
    cleaned_output_folder = input("Inserisci il percorso della cartella cleaned_output_folder: ").strip()
    
    # Chiedi la durata degli slot temporali
    try:
        time_slot_minutes = int(input("Inserisci la durata degli slot temporali in minuti (default 30): ") or "30")
    except ValueError:
        time_slot_minutes = 30
    
    success = create_plots_for_dataset(cleaned_output_folder, time_slot_minutes=time_slot_minutes)
    if not success:
        print("Operazione fallita!")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())