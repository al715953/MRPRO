import pandas as pd
import numpy as np
from itertools import product

def run_sniper_calibration(csv_path='src/data/Melate-Retro.csv'):
    df = pd.read_csv(csv_path).sort_values('CONCURSO', ascending=True)
    winning_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    history = df[winning_cols].values
    
    test_window = 1000
    start_idx = len(history) - test_window
    
    # Pesos de alta fidelidad detectados previamente
    w_gap, w_term, w_freq = 0.25, 0.10, 0.60
    # Umbral de Sniper (Ajustable para buscar el 95%+)
    thresholds = [0.80, 0.85, 0.90, 0.95]
    
    print(f"🎯 Calibrando Sniper sobre {test_window} sorteos...")
    
    for tau in thresholds:
        kills = 0
        draws_active = 0
        
        for i in range(start_idx, len(history)):
            past = history[:i]
            actual = set(history[i])
            
            # Cálculo de Señales
            gaps = np.full(40, i)
            for g_v, draw in enumerate(reversed(past)):
                for n in draw:
                    if gaps[int(n)] == i: gaps[int(n)] = g_v
            max_g = np.max(gaps[1:])
            
            last_10 = past[-10:].flatten()
            t_counts = np.bincount(last_10.astype(int) % 10, minlength=10)
            
            last_50 = past[-50:].flatten()
            f_counts = np.bincount(last_50.astype(int), minlength=41)
            
            # Scoring
            scores = []
            for n in range(1, 40):
                s = (gaps[n]/max_g)*w_gap + (1.0 if t_counts[n%10]>3 else 0)*w_term + (1.0 if f_counts[n]>8 else 0)*w_freq
                scores.append(s)
            
            max_val = max(scores)
            voted_num = np.argmax(scores) + 1
            
            # --- LÓGICA SNIPER ---
            if max_val >= tau:
                draws_active += 1
                if voted_num in actual:
                    kills += 1
        
        acc = (draws_active - kills) / draws_active * 100 if draws_active > 0 else 0
        covertura = (draws_active / test_window) * 100
        
        print(f"Umbral {tau:.2f} -> Precisión: {acc:.2f}% | Se activa en: {covertura:.1f}% de sorteos")

if __name__ == "__main__":
    run_sniper_calibration()