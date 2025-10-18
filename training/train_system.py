# causal_swarm_poc/training/train_system.py

import sys
import os
import torch
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fact_extractor import train as train_fact_extractor # Importa a função de treinamento
from models.hopfield_network import HopfieldNetwork, bipolarize
# from models.causal_reasoner import create_reasoner_model_and_tokenizer
# from models.causal_reasoner import format_reasoner_prompt
# from micro_world.causal_graph import compute_interventional_probability

# --- Configurações Globais ---
CORPUS_PATH = "data/corpus.csv"
FACT_EXTRACTOR_OUTPUT_DIR = "models/fact_extractor_finetuned"
REASONER_OUTPUT_DIR = "models/causal_reasoner_finetuned"

def train_hopfield() -> HopfieldNetwork:
    """
    Etapa 2: Treina a Rede de Hopfield com os padrões causais definidos.
    Na nossa PoC, este passo é fixo e não depende do LLM1.
    """
    print("\n--- Etapa 2: Treinando a Rede de Hopfield ---")
    
    true_dag_binary = np.array([1, 0, 1, 1, 0, 1]) # [C->X, C->Y, C->Z, X->Y, X->Z, Y->Z]
    wrong_dag_binary = np.array([1, 0, 1, 0, 1, 0])
    patterns_bipolar = [bipolarize(true_dag_binary), bipolarize(wrong_dag_binary)]
    
    hopfield_net = HopfieldNetwork(num_neurons=len(true_dag_binary))
    hopfield_net.train(patterns_bipolar)
    
    return hopfield_net

def train_causal_reasoner(corpus_df: pd.DataFrame, hopfield_net: HopfieldNetwork) -> None:
    """
    Etapa 3: Define a lógica de treinamento para o Raciocinador Causal (LLM2).
    """
    print("\n--- Etapa 3: Definindo a lógica de treinamento para o Raciocinador Causal (LLM2) ---")
    
    # Esta parte permanece conceitual, como discutido anteriormente.
    # A implementação completa requereria uma subclasse do Trainer do Hugging Face.
    
    print("Conceito de Treinamento do LLM2 definido.")
    print("A implementação completa do treinamento do LLM2 está fora do escopo desta PoC inicial.")
    print("Foco: 1. Treinar LLM1. 2. Estruturar Hopfield. 3. Definir arquitetura LLM2.")

if __name__ == '__main__':
    print("--- Orquestrador de Treinamento do Sistema Causal ---")
    
    if not os.path.exists(CORPUS_PATH):
        print(f"Erro: Arquivo de corpus não encontrado em '{CORPUS_PATH}'.")
        print("Execute 'python data_generation/text_corpus_generator.py' primeiro.")
    else:
        # --- Etapa 1: Treinar o Extrator de Fatos (LLM1) ---
        print("\n--- Etapa 1: Treinando o Extrator de Fatos (LLM1) ---")
        # Chama a função de treinamento diretamente do módulo fact_extractor
        train_fact_extractor(corpus_path=CORPUS_PATH)
        print("\nLLM1 (Extrator de Fatos) treinado com sucesso.")

        # --- Etapa 2: Treinar a Rede de Hopfield ---
        # Neste ponto, o LLM1 está treinado. A Hopfield Net pode ser treinada.
        hopfield_network = train_hopfield()
        
        # --- Etapa 3: Definir a lógica para o Raciocinador Causal (LLM2) ---
        corpus_df = pd.read_csv(CORPUS_PATH)
        train_causal_reasoner(corpus_df, hopfield_network)