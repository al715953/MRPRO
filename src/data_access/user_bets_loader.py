import pandas as pd
from typing import List
from src.domain.dtos import UserBetDTO

class UserBetsLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_bets(self) -> List[UserBetDTO]:
        bets = []
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip().str.upper()
            
            # Busca columnas fecha y numeros (N1...N6)
            cols_nums = [c for c in df.columns if c.startswith('N')][:6]
            
            for _, row in df.iterrows():
                try:
                    fecha = pd.to_datetime(row['FECHA'], dayfirst=True).date()
                    nums = [int(row[c]) for c in cols_nums]
                    costo = float(row.get('COSTO', 10.0))
                    
                    bets.append(UserBetDTO(fecha, nums, costo))
                except Exception as e:
                    continue # Saltar fila con error
            
            return bets
        except Exception as e:
            print(f"⚠️ No se pudo cargar historial de usuario: {e}")
            return []