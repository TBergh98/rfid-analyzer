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
# Nuove importazioni per il grafo
import networkx as nx

def create_copresence_network_plot(df, output_folder, copresence_threshold=1):
    """
    Crea la matrice di copresenza GxG e il network plot delle galline.
    Args:
        df: DataFrame con colonne ['data', 'ora', 'comportamento', 'id_gallina', 'id_nido']
        output_folder: cartella di output
        copresence_threshold: soglia minima per visualizzare un arco
    """
    # Filtra solo eventi IN e OUT
    df = df[df['comportamento'].isin(['IN', 'OUT'])].copy()
    df['datetime'] = pd.to_datetime(df['data'] + ' ' + df['ora'], format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values(['id_nido', 'id_gallina', 'datetime'])

    # Costruisci intervalli di presenza per ogni gallina in ogni nido
    intervals = []
    for nido, group in df.groupby('id_nido'):
        for gallina, ggroup in group.groupby('id_gallina'):
            # Trova intervalli IN/OUT
            in_times = ggroup[ggroup['comportamento'] == 'IN']['datetime'].tolist()
            out_times = ggroup[ggroup['comportamento'] == 'OUT']['datetime'].tolist()
            # Associa ogni IN al primo OUT successivo
            idx_in = 0
            idx_out = 0
            while idx_in < len(in_times):
                in_time = in_times[idx_in]
                # Trova il primo OUT dopo IN
                out_time = None
                while idx_out < len(out_times) and out_times[idx_out] <= in_time:
                    idx_out += 1
                if idx_out < len(out_times):
                    out_time = out_times[idx_out]
                    idx_out += 1
                else:
                    out_time = in_time + pd.Timedelta(minutes=10)  # fallback: 10 min
                intervals.append({'id_nido': nido, 'id_gallina': gallina, 'in_time': in_time, 'out_time': out_time})
                idx_in += 1

    # Costruisci matrice di copresenza
    galline = sorted(df['id_gallina'].unique())
    gallina_idx = {g: i for i, g in enumerate(galline)}
    copresence_matrix = np.zeros((len(galline), len(galline)), dtype=int)

    # Per ogni nido, trova copresenze tra galline
    from itertools import combinations
    intervals_by_nido = {}
    for interval in intervals:
        intervals_by_nido.setdefault(interval['id_nido'], []).append(interval)
    for nido, nido_intervals in intervals_by_nido.items():
        # Per ogni coppia di galline
        for i1, i2 in combinations(nido_intervals, 2):
            g1, g2 = i1['id_gallina'], i2['id_gallina']
            # Sovrapposizione intervalli
            latest_start = max(i1['in_time'], i2['in_time'])
            earliest_end = min(i1['out_time'], i2['out_time'])
            overlap = (earliest_end - latest_start).total_seconds()
            if overlap > 0:
                copresence_matrix[gallina_idx[g1], gallina_idx[g2]] += 1
                copresence_matrix[gallina_idx[g2], gallina_idx[g1]] += 1

    # Salva la matrice copresenza
    copresence_df = pd.DataFrame(copresence_matrix, index=galline, columns=galline)
    copresence_path = os.path.join(output_folder, 'copresence_matrix.csv')
    copresence_df.to_csv(copresence_path)
    print(f"Matrice di copresenza salvata in: {copresence_path}")

    # Costruisci grafo e posizioni nodi con NetworkX (solo per layout)
    G = nx.Graph()
    for i, g1 in enumerate(galline):
        G.add_node(g1)
    for i in range(len(galline)):
        for j in range(i+1, len(galline)):
            weight = copresence_matrix[i, j]
            if weight >= copresence_threshold:
                G.add_edge(galline[i], galline[j], weight=weight)

    pos = nx.spring_layout(G, seed=42, k=1.5)
    # Prepara dati per Plotly
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_text = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(d['weight'])
        edge_text.append(f'{u} - {v}: {d["weight"]}')

    # Colori archi in base al peso (YlGnBu: più blu = più copresenze)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=min(edge_weights) if edge_weights else 0, vmax=max(edge_weights) if edge_weights else 1)
    cmap = cm.get_cmap('YlGnBu')
    edge_colors = [f'rgba{cmap(norm(w))[:3] + (0.7,)}' for w in edge_weights]

    # Plot archi
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Plot nodi
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    # Colora i nodi in base al grado
    node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_adjacencies,
            size=12,
            colorbar=dict(
                thickness=15,
                title='Grado nodo',
                xanchor='left',
            )
        )
    )

    # Crea la figura
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Grafo Interattivo delle Galline (copresenza ≥ {copresence_threshold})',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="<b>Galline</b>: nodo = gallina, arco = copresenza, tooltip = peso",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )

    # Tooltip sugli archi (aggiunti come hovertext su edge_trace)
    # Plotly non supporta hovertext su lines multiple, workaround: aggiungi archi come scatter separati
    edge_traces = []
    for (u, v, d), color in zip(G.edges(data=True), edge_colors):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=color),
            hoverinfo='text',
            text=[f'{u} - {v}: {d["weight"]}', f'{u} - {v}: {d["weight"]}'],
            mode='lines+markers',
            marker=dict(size=18, color=color, opacity=0.5, line=dict(width=0))
        ))
    # Aggiungi una colorbar custom per gli archi (copresenze)
    if edge_weights:
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale='YlGnBu',
                cmin=0,
                cmax=max(edge_weights),
                color=[max(edge_weights)],
                colorbar=dict(
                    title='Copresenze arco',
                    thickness=15,
                    x=1.05,
                    y=0.5,
                    len=0.7,
                    tickvals=[0, max(edge_weights)],
                    ticktext=[str(0), str(max(edge_weights))]
                ),
                showscale=True
            ),
            showlegend=False
        )
        fig = go.Figure(data=edge_traces + [node_trace, colorbar_trace], layout=fig.layout)
    else:
        fig = go.Figure(data=edge_traces + [node_trace], layout=fig.layout)

    # Determina il gruppo di nidi dal DataFrame
    nest_ids = set(df['id_nido'].apply(lambda x: str(int(float(x))) if isinstance(x, (int, float)) else str(x)))
    if nest_ids.issubset(set(str(i) for i in range(11, 15))):
        group_label = 'nidi_1.1-1.4'
    elif nest_ids.issubset(set(str(i) for i in range(21, 25))):
        group_label = 'nidi_2.1-2.4'
    else:
        group_label = 'nidi_misti'
    plot_path = os.path.join(output_folder, f'copresence_network_interactive_{group_label}.html')
    plot(fig, filename=plot_path, auto_open=False)
    print(f"Network plot interattivo salvato in: {plot_path}")

def select_subfolder(cleaned_output_folder):
    """Permette all'utente di selezionare una sottocartella threshold_<pre>s_<post>s."""
    # Filtra solo le cartelle che iniziano con "threshold_" e contengono due soglie
    subfolders = [f for f in os.listdir(cleaned_output_folder)
                  if os.path.isdir(os.path.join(cleaned_output_folder, f))
                  and f.startswith('threshold_') and 's_' in f and f.endswith('s')]
    if not subfolders:
        print("Nessuna sottocartella 'threshold_<pre>s_<post>s' trovata!")
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
    """Crea due cluster map interattive con Plotly: una per i nidi 1.1-1.4, una per i nidi 2.1-2.4."""
    # Filtra solo gli ingressi nei nidi
    entries = df[df['comportamento'] == 'IN'].copy()
    entries['formatted_nest_id'] = entries['id_nido'].apply(format_nest_id)
    # Definisci i gruppi di nidi
    group1_nests = [f"1.{i}" for i in range(1, 5)]
    group2_nests = [f"2.{i}" for i in range(1, 5)]
    for group_nests, group_label in [(group1_nests, "nidi_1.1-1.4"), (group2_nests, "nidi_2.1-2.4")]:
        group_entries = entries[entries['formatted_nest_id'].isin(group_nests)].copy()
        visit_matrix = group_entries.groupby(['id_gallina', 'formatted_nest_id']).size().unstack(fill_value=0)
        if visit_matrix.empty:
            print(f"Nessun dato di ingresso trovato per creare la cluster map per {group_label}.")
            continue
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(visit_matrix.values)
        n_clusters = min(5, len(visit_matrix))  # Fino a 5 cluster
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_data)
        else:
            cluster_labels = [0] * len(visit_matrix)
        visit_matrix_sorted = visit_matrix.copy()
        visit_matrix_sorted['cluster'] = cluster_labels
        visit_matrix_sorted = visit_matrix_sorted.sort_values('cluster')
        visit_matrix_sorted = visit_matrix_sorted.drop('cluster', axis=1)
        fig = go.Figure(data=go.Heatmap(
            z=visit_matrix_sorted.values,
            x=visit_matrix_sorted.columns,
            y=[f"Gallina {idx}" for idx in visit_matrix_sorted.index],
            colorscale='Viridis',
            hovertemplate='Gallina: %{y}<br>Nido: %{x}<br>Visite: %{z}<extra></extra>',
            colorbar=dict(title="Numero di Visite")
        ))
        fig.update_layout(
            title=f'Cluster Map - Frequentazione dei Nidi per Gallina ({group_label})',
            xaxis_title='ID Nido',
            yaxis_title='Galline',
            width=800,
            height=600,
            font=dict(size=12)
        )
        plot_path = os.path.join(output_folder, f'clustering_heatmap_{group_label}.html')
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

def create_plots_for_dataset(
cleaned_output_folder,
selected_subfolder=None,
time_slot_minutes=30,
plot_day_window=1,
egg_laying_start_day=None,
plot_nest_preference=True,
plot_cluster_heatmap=True,
plot_timeflows=True,
plot_network_copresence=True,
copresence_thresholds=None
):
    """
    Funzione principale per creare i plots. 
    Può essere chiamata da un main esterno.
    
    Args:
        cleaned_output_folder (str): Percorso alla cartella principale
        selected_subfolder (str, optional): Nome della sottocartella specifica. 
                                          Se None, prompta l'utente per la selezione.
        time_slot_minutes (int): Durata degli slot temporali in minuti
        plot_day_window (int): Finestra temporale in giorni
        egg_laying_start_day (str): Data di inizio ovodeposizione (YYYY-MM-DD)
    
    Returns:
        bool: True se i plots sono stati creati con successo, False altrimenti
    """
    import shutil
    if not os.path.exists(cleaned_output_folder):
        print(f"Il percorso specificato non esiste: {cleaned_output_folder}")
        return False
    
    # Seleziona la sottocartella
    if selected_subfolder:
        folder_path = os.path.join(cleaned_output_folder, selected_subfolder)
        if not os.path.exists(folder_path):
            print(f"Sottocartella specificata non trovata: {folder_path}")
            return False
        selected_folder = folder_path
    else:
        selected_folder = select_subfolder(cleaned_output_folder)
        if not selected_folder:
            return False
    print(f"Cartella selezionata: {selected_folder}")
    
    # Carica i dati CSV direttamente dalla cartella threshold_<seconds>s
    df = load_csv_data(selected_folder)
    if df is None or df.empty:
        print("Nessun dato trovato per la generazione dei plot.")
        return False
    print(f"Caricati {len(df)} record dai file CSV")
    
    # Parsing date
    df['datetime'] = pd.to_datetime(df['data'] + ' ' + df['ora'], format='%d/%m/%Y %H:%M:%S')
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    if not egg_laying_start_day:
        print("egg_laying_start_day non specificato, verrà usato l'intero periodo.")
        periods = [(min_date, max_date)]
    else:
        egg_start = pd.to_datetime(egg_laying_start_day).date()
        if egg_start <= min_date or egg_start > max_date:
            print("egg_laying_start_day fuori dal range dei dati, verrà usato l'intero periodo.")
            periods = [(min_date, max_date)]
        else:
            periods = [(min_date, egg_start - pd.Timedelta(days=1)), (egg_start, max_date)]
    # Usa direttamente i parametri booleani passati

    success = True
    for idx, (start, end) in enumerate(periods):
        folder_name = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
        period_folder = os.path.join(selected_folder, folder_name)
        os.makedirs(period_folder, exist_ok=True)
        period_df = df[(df['datetime'].dt.date >= start) & (df['datetime'].dt.date <= end)].copy()
        if period_df.empty:
            print(f"Nessun dato per il periodo {folder_name}, salto.")
            continue
        plots_folder = os.path.join(period_folder, 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        try:
            if plot_nest_preference:
                print(f"[Periodo {folder_name}] Creazione del bar plot delle preferenze dei nidi...")
                create_nest_preference_plot(period_df, plots_folder)
            if plot_cluster_heatmap:
                print(f"[Periodo {folder_name}] Creazione della cluster map interattiva...")
                create_clustering_heatmap(period_df, plots_folder)
            if plot_timeflows:
                print(f"[Periodo {folder_name}] Creazione del plot temporale dei flussi (slot di {time_slot_minutes} minuti, finestra di {plot_day_window} giorni)...")
                create_time_slot_flow_plot(period_df, plots_folder, time_slot_minutes, plot_day_window)
            if plot_network_copresence:
                # Determina il periodo (pre o post egg laying)
                if len(periods) == 1:
                    period_key = 'pre_egg_laying'  # fallback se solo un periodo
                else:
                    period_key = 'pre_egg_laying' if idx == 0 else 'post_egg_laying'
                thresholds = copresence_thresholds.get(period_key, {}) if copresence_thresholds else {}
                print(f"[Periodo {folder_name}] Creazione network plot copresenza per nidi 1.1-1.4...")
                group1_nests = [f"1.{i}" for i in range(1, 5)]
                group1_df = period_df[period_df['id_nido'].apply(lambda x: str(int(float(x))) if isinstance(x, (int, float)) else str(x)).isin([str(i) for i in range(11, 15)])].copy()
                group1_threshold = thresholds.get('group1', 1)
                if not group1_df.empty:
                    create_copresence_network_plot(group1_df, plots_folder, copresence_threshold=group1_threshold)
                else:
                    print("Nessun dato per nidi 1.1-1.4 in questo periodo.")

                print(f"[Periodo {folder_name}] Creazione network plot copresenza per nidi 2.1-2.4...")
                group2_nests = [f"2.{i}" for i in range(1, 5)]
                group2_df = period_df[period_df['id_nido'].apply(lambda x: str(int(float(x))) if isinstance(x, (int, float)) else str(x)).isin([str(i) for i in range(21, 25)])].copy()
                group2_threshold = thresholds.get('group2', 1)
                if not group2_df.empty:
                    create_copresence_network_plot(group2_df, plots_folder, copresence_threshold=group2_threshold)
                else:
                    print("Nessun dato per nidi 2.1-2.4 in questo periodo.")
        except Exception as e:
            print(f"Errore durante la creazione dei plots per il periodo {folder_name}: {e}")
            success = False
    return success

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