# causal_swarm_poc/training/train_system.py

import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedModel, 
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from peft import PeftModel
from typing import cast, Dict, Tuple

# Adiciona o diretório raiz ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fact_extractor import extract_facts
from models.hopfield_network import HopfieldNetwork, bipolarize, debipolarize
from models.causal_reasoner import format_reasoner_prompt
from micro_world.causal_graph import compute_interventional_probability

# --- Configurações Globais ---
FACT_EXTRACTOR_PATH = "models/fact_extractor_finetuned"
REASONER_OUTPUT_DIR = "models/causal_reasoner_finetuned"
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
NUM_EDGES = 6 # [C->X, C->Y, C->Z, X->Y, X->Z, Y->Z]

def load_fact_extractor() -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Carrega o modelo LLM1 fine-tuned."""
    print("Carregando o Extrator de Fatos (LLM1)...")
    base_model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto"
    ))
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    ))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, FACT_EXTRACTOR_PATH)
    model = model.eval()
    return model, tokenizer

def train_hopfield(corpus_df: pd.DataFrame, llm1: PeftModel, tokenizer: PreTrainedTokenizer) -> HopfieldNetwork:
    """
    Passo 1: Extrai fatos do corpus e usa-os para treinar a Rede de Hopfield.
    """
    print("\n--- Etapa 1: Treinando a Rede de Hopfield ---")
    
    # Extrai a estrutura causal de cada linha de texto
    extracted_dags = []
    print("Extraindo DAGs implícitos do corpus usando LLM1...")
    for text in tqdm(corpus_df['text']):
        facts = extract_facts(text, llm1, tokenizer)
        if facts:
            # Simplificação: assumimos que podemos derivar uma estrutura de DAG
            # a partir dos fatos. Na nossa PoC, isso é um passo simulado.
            # Vamos apenas usar o DAG verdadeiro + um DAG falso como padrões.
            pass

    # Na PoC, em vez de extrair, vamos 'ensinar' a rede com os padrões que queremos.
    true_dag_binary = np.array([1, 0, 1, 1, 0, 1])
    wrong_dag_binary = np.array([1, 0, 1, 0, 1, 0])
    patterns_bipolar = [bipolarize(true_dag_binary), bipolarize(wrong_dag_binary)]
    
    hopfield_net = HopfieldNetwork(num_neurons=NUM_EDGES)
    hopfield_net.train(patterns_bipolar)
    
    return hopfield_net

def train_causal_reasoner(corpus_df: pd.DataFrame, hopfield_net: HopfieldNetwork):
    """
    Passo 2: Treina o Raciocinador Causal (LLM2) usando a função de custo híbrida.
    Esta é a parte mais complexa e aqui é apresentada de forma simplificada.
    """
    print("\n--- Etapa 2: Treinando o Raciocinador Causal (LLM2) ---")
    
    # Carrega o modelo base para o LLM2
    # A implementação real usaria o Trainer do Hugging Face com uma subclasse customizada.
    
    # 1. Definir o Dataset de Treinamento
    # Cada item do dataset será uma pergunta, e a resposta correta.
    # Ex: Pergunta: "Qual o efeito causal do tratamento?"
    #     Resposta: "O tratamento aumenta a probabilidade de recuperação."

    # 2. Criar uma subclasse do Trainer para a Perda Híbrida
    class CausalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Perda linguística padrão (cross-entropy)
            loss_lm, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # --- Início da Lógica da Perda Causal ---
            # Esta parte é a inovação.
            
            # Exemplo de como poderia funcionar:
            # Para cada lote, fazemos uma pergunta intervencional.
            question = "Qual a probabilidade de recuperação se aplicarmos o tratamento?"
            
            # Usamos o estado da rede de Hopfield como contexto
            # (Simplificação: usamos o DAG verdadeiro como a 'crença' correta)
            true_dag_binary = np.array([1, 0, 1, 1, 0, 1])
            prompt = format_reasoner_prompt(question, true_dag_binary)
            
            # Geramos uma resposta com o modelo atual
            # ... a lógica de geração aqui ...
            generated_text = "A probabilidade de recuperação é 75%." # Exemplo de saída da IA
            
            # Extraímos o valor numérico da resposta
            predicted_prob = 0.75 # Lógica de parsing para extrair o número
            
            # Obtemos a verdade fundamental do nosso micro-mundo
            true_prob = compute_interventional_probability(do_x=1)
            
            # Calculamos a perda causal (ex: Erro Quadrático Médio)
            loss_causal = (predicted_prob - true_prob) ** 2
            
            # Ponderamos as perdas
            lambda_causal = 0.1 # Hiperparâmetro
            total_loss = loss_lm + lambda_causal * loss_causal
            
            return (total_loss, outputs) if return_outputs else total_loss

    print("Conceito de Treinamento do LLM2 definido.")
    print("A implementação completa requer uma subclasse do Trainer e um loop de treinamento detalhado.")
    print("Este script demonstra a arquitetura, mas não executa o treinamento do LLM2.")
    

if __name__ == '__main__':
    print("--- Orquestrador de Treinamento do Sistema Causal ---")
    
    # Carregar o corpus de dados
    corpus_path = "data/corpus.csv"
    if not os.path.exists(corpus_path):
        print(f"Erro: Arquivo de corpus não encontrado em '{corpus_path}'.")
        print("Execute 'python data_generation/text_corpus_generator.py' primeiro.")
    else:
        corpus_df = pd.read_csv(corpus_path)
        
        # Carregar o LLM1 (já treinado)
        llm1, tokenizer_llm1 = load_fact_extractor()

        # Etapa 1: Usar os dados para treinar a Rede de Hopfield
        hopfield_network = train_hopfield(corpus_df, llm1, tokenizer_llm1)
        
        # Etapa 2: Definir a lógica de treinamento para o Raciocinador Causal (LLM2)
        train_causal_reasoner(corpus_df, hopfield_network)