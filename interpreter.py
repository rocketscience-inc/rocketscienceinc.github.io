import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# ------------------------------------------------------------
# Configuração de logs
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------------------------------------------
# Carregamento do modelo
# ------------------------------------------------------------
def load_model(model_path: str):
    """
    Carrega o modelo treinado em formato .pkl.
    """
    logging.info(f"Carregando modelo: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError("Arquivo .pkl não encontrado.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logging.info("Modelo carregado com sucesso.")
    return model


# ------------------------------------------------------------
# Pré-processamento dos dados
# ------------------------------------------------------------
def preprocess(data: pd.DataFrame):
    """
    Aplica transformações necessárias antes da predição.
    Aqui entram normalizações, remoção de outliers, etc.
    """
    logging.info("Iniciando pré-processamento dos dados...")

    # Exemplo fictício de normalização
    df = data.copy()
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    logging.info("Pré-processamento concluído.")
    return df


# ------------------------------------------------------------
# Predição 
# ------------------------------------------------------------
def predict_rul(model, data: pd.DataFrame):
    """
    Gera a previsão de RUL utilizando o modelo carregado.
    Limita o número máximo de ciclos em 200.
    """
    logging.info("Executando predição...")

    processed = preprocess(data)
    pred = model.predict(processed)

    # Limita o máximo do ciclo
    pred = np.clip(pred, 0, 200)

    logging.info("Predição finalizada.")
    return pred


# ------------------------------------------------------------
# Função principal
# ------------------------------------------------------------
def run_prediction(model_path: str, csv_path: str):
    """
    Função principal que controla todo o fluxo:
    - Carrega modelo
    - Lê arquivo .csv
    - Realiza predições
    """
    logging.info("Iniciando pipeline de previsão.")

    model = load_model(model_path)

    logging.info(f"Lendo dados de entrada: {csv_path}")
    data = pd.read_csv(csv_path)

    predictions = predict_rul(model, data)

    logging.info("Pipeline concluído.")
    return predictions


# ------------------------------------------------------------
# Execução direta
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        resultado = run_prediction(
            model_path="modelo_treinado.pkl",
            csv_path="dados_novos.csv"
        )

        print("\n=== PREVISÕES DE RUL E CYCLES MAX ===")
        print(resultado)

    except Exception as e:
        logging.error(f"Erro ao executar o interpretador: {e}")
