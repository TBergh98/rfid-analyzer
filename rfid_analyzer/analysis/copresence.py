"""Copresence analysis for chicken RFID data."""

import os
from itertools import combinations
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd


def create_copresence_matrix(
    csv_path: Union[str, Path],
    output_folder: Union[str, Path],
    mode: Literal["count", "duration"] = "count",
) -> pd.DataFrame:
    """
    Create copresence matrix (GxG) analyzing chicken copresence in nests.
    
    Args:
        csv_path: Path to cleaned CSV file with columns ['Data', 'Ora', 'Azione', 'ID Gallina', 'ID Nido']
        output_folder: Output folder for CSV matrix
        mode: 'count' counts number of copresence events, 'duration' sums seconds of copresence
    
    Returns:
        DataFrame with copresence matrix (chickens x chickens)
    """
    csv_path = Path(csv_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Column mapping: Italian names to expected names
    column_mapping = {
        'Data': 'data',
        'Ora': 'ora',
        'Azione': 'comportamento',
        'ID Gallina': 'id_gallina',
        'ID Nido': 'id_nido'
    }
    df = df.rename(columns=column_mapping)
    
    # Filter only IN and OUT events
    df = df[df['comportamento'].isin(['IN', 'OUT'])].copy()
    df['datetime'] = pd.to_datetime(df['data'] + ' ' + df['ora'], format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values(['id_nido', 'id_gallina', 'datetime'])
    
    # Build presence intervals for each chicken in each nest
    intervals = []
    for nido, group in df.groupby('id_nido'):
        for gallina, ggroup in group.groupby('id_gallina'):
            # Find IN/OUT intervals
            in_times = ggroup[ggroup['comportamento'] == 'IN']['datetime'].tolist()
            out_times = ggroup[ggroup['comportamento'] == 'OUT']['datetime'].tolist()
            
            # Associate each IN with the first following OUT
            idx_in = 0
            idx_out = 0
            while idx_in < len(in_times):
                in_time = in_times[idx_in]
                # Find the first OUT after IN
                out_time = None
                while idx_out < len(out_times) and out_times[idx_out] <= in_time:
                    idx_out += 1
                if idx_out < len(out_times):
                    out_time = out_times[idx_out]
                    idx_out += 1
                else:
                    out_time = in_time + pd.Timedelta(minutes=10)  # fallback: 10 min
                intervals.append({
                    'id_nido': nido,
                    'id_gallina': gallina,
                    'in_time': in_time,
                    'out_time': out_time
                })
                idx_in += 1
    
    # Build copresence matrix
    galline = sorted(df['id_gallina'].unique())
    gallina_idx = {g: i for i, g in enumerate(galline)}
    
    # Use float for duration mode, int for count mode
    dtype = float if mode == 'duration' else int
    copresence_matrix = np.zeros((len(galline), len(galline)), dtype=dtype)
    
    # For each nest, find copresences between chickens
    intervals_by_nido = {}
    for interval in intervals:
        intervals_by_nido.setdefault(interval['id_nido'], []).append(interval)
    
    for nido, nido_intervals in intervals_by_nido.items():
        # For each pair of chickens
        for i1, i2 in combinations(nido_intervals, 2):
            g1, g2 = i1['id_gallina'], i2['id_gallina']
            # Interval overlap
            latest_start = max(i1['in_time'], i2['in_time'])
            earliest_end = min(i1['out_time'], i2['out_time'])
            # Convert to timedelta and get seconds
            overlap_td = earliest_end - latest_start
            overlap = overlap_td.total_seconds() if hasattr(overlap_td, 'total_seconds') else overlap_td / pd.Timedelta(seconds=1)
            
            if overlap > 0:
                if mode == 'duration':
                    # Accumulate seconds of copresence
                    copresence_matrix[gallina_idx[g1], gallina_idx[g2]] += overlap
                    copresence_matrix[gallina_idx[g2], gallina_idx[g1]] += overlap
                else:  # mode == 'count'
                    # Count number of copresence events
                    copresence_matrix[gallina_idx[g1], gallina_idx[g2]] += 1
                    copresence_matrix[gallina_idx[g2], gallina_idx[g1]] += 1
    
    # Create DataFrame
    copresence_df = pd.DataFrame(copresence_matrix, index=galline, columns=galline)
    
    # Determine output filename with mode suffix
    base_name = csv_path.stem  # e.g., 'pre_egg_laying_group1'
    mode_suffix = f"_{mode}"
    output_filename = f'copresence_matrix{mode_suffix}_{base_name}.csv'
    output_path = output_folder / output_filename
    
    # Save matrix
    copresence_df.to_csv(output_path)
    
    # Print summary
    if mode == 'duration':
        # Convert to minutes for display
        total_seconds = copresence_matrix.sum() / 2  # Divide by 2 because matrix is symmetric
        total_minutes = total_seconds / 60
        print(f"Copresence matrix ({mode}) saved to: {output_path}")
        print(f"Total copresence time: {total_seconds:.1f} seconds ({total_minutes:.1f} minutes)")
    else:
        total_events = int(copresence_matrix.sum() / 2)
        print(f"Copresence matrix ({mode}) saved to: {output_path}")
        print(f"Total copresence events: {total_events}")
    
    return copresence_df
