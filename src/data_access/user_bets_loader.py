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
            
            # Asumimos columnas: FECHA, N1, N2, N3, N4, N5, N6
            # Opcional: COSTO
            
            for _, row in df.iterrows():
                # Obtener fecha
                bet_date = pd.to_datetime(row['FECHA'], dayfirst=True).date()
                
                # Obtener números (buscamos N1..N6)
                ticket = [
                    int(row['N1']), int(row['N2']), int(row['N3']),
                    int(row['N4']), int(row['N5']), int(row['N6'])
                ]
                
                # Obtener costo (si existe, sino 10.0 por defecto)
                cost = float(row['COSTO']) if 'COSTO' in df.columns else 10.0
                
                bets.append(UserBetDTO(
                    date=bet_date,
                    ticket_numbers=ticket,
                    cost=cost
                ))
                
            print(f"✅ Apuestas personales cargadas: {len(bets)} registros.")
            return bets

        except FileNotFoundError:
            print(f"⚠️ Archivo de apuestas no encontrado: {self.csv_path}")
            return []
        except Exception as e:
            print(f"❌ Error leyendo apuestas: {e}")
            return []