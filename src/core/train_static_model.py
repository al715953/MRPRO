# src/core/train_static_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Rutas Relativas (Asumiendo ejecución desde la raíz del proyecto)
# El script se ejecuta como módulo o desde raíz, pero ajustamos para robustez.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "src", "data", "Melate-Retro.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "src", "data","mrpro_model_v8_static.json")
TOTAL_BALLS = 39

def train_master_brain():
    print(f"🧠 INICIANDO ENTRENAMIENTO MAESTRO (V8 Static) desde Core...")
    print(f"   📂 Buscando datos en: {DATA_FILE}")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: No encuentro el archivo de datos.")
        # Intento de fallback
        fallback = "Melate-Retro.csv"
        if os.path.exists(fallback):
            print(f"   ⚠️ Encontrado en raíz, usándolo.")
            df = pd.read_csv(fallback)
        else:
            return
    else:
        df = pd.read_csv(DATA_FILE)

    print(f"📚 Historial cargado: {len(df)} sorteos.")

    # 1. PREPARACIÓN DE DATOS (Dataset Masivo)
    real_draws = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6']].values.astype(np.uint8)
    
    # Generación de Ruido (Ratio 1:10 para V8 - Más exigente)
    n_real = len(real_draws)
    n_fake = n_real * 10  # Aumentamos la dificultad para que el modelo sea más crítico
    fake_draws = np.random.randint(1, TOTAL_BALLS + 1, size=(n_fake, 6)).astype(np.uint8)
    fake_draws.sort(axis=1)

    X = np.vstack([real_draws, fake_draws])
    y = np.hstack([np.ones(n_real), np.zeros(n_fake)])

    # 2. ENTRENAMIENTO PROFUNDO
    print(f"🏋️ Entrenando con {len(X)} patrones (Profundidad 12, 1000 Rondas)...")
    
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist", 
        "eval_metric": "logloss",
        "max_depth": 12,       # Más profundo = Más memoria
        "eta": 0.02,           # Aprendizaje ultra-lento = Más precisión
        "subsample": 0.9,
        "colsample_bytree": 0.9
    }
    
    # 1000 Rondas (Entrenamiento intensivo)
    bst = xgb.train(params, dtrain, num_boost_round=1000)
    
    # 3. GUARDADO
    bst.save_model(MODEL_OUTPUT)
    print(f"✅ CEREBRO V8 GUARDADO EN: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_master_brain()