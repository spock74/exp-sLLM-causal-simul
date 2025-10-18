# causal_swarm_poc/data_generation/text_corpus_generator.py

import sys
import os
import argparse
import pandas as pd
from typing import Dict, List, Union

# Adiciona o diretório raiz ao path para que possamos importar de 'micro_world'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from micro_world.causal_graph import sample

# --- Templates de Texto ---

# Mapeamentos de valores binários para texto descritivo
C_MAP: Dict[int, str] = {0: "nenhuma condição pré-existente notável", 1: "uma condição pré-existente relevante"}
X_MAP: Dict[int, str] = {0: "placebo", 1: "droga experimental"}
Y_MAP: Dict[int, str] = {0: "nível normal de biomarcador", 1: "nível elevado de biomarcador"}
Z_MAP: Dict[int, str] = {0: "uma recuperação completa", 1: "não apresentou recuperação"}

# Type aliases for clarity
PatientData = Dict[str, int]
FormattedEntry = Dict[str, Union[str, int]]

def format_observational(patient_id: int, data_point: PatientData) -> FormattedEntry:
    """Formata um ponto de dados como uma frase de relatório observacional."""
    text = (
        f"Relatório do Paciente {patient_id}: O histórico do paciente indica "
        f"{C_MAP[data_point['C']]}. Foi administrado {X_MAP[data_point['X']]}. "
        f"A análise subsequente mostrou um {Y_MAP[data_point['Y']]}. "
        f"O desfecho observado foi {Z_MAP[data_point['Z']]}."
    )
    return {
        "text": text,
        "type": "observational",
        **data_point # Adiciona as colunas C, X, Y, Z
    }

def format_interventional(trial_id: int, data_point: PatientData) -> FormattedEntry:
    """Formata um ponto de dados como uma frase de ensaio clínico (intervenção)."""
    text = (
        f"Registro do Estudo Clínico RCT{trial_id}: O participante, que apresentava "
        f"{C_MAP[data_point['C']]}, foi randomizado para receber {X_MAP[data_point['X']]}. "
        f"A medição do biomarcador indicou {Y_MAP[data_point['Y']]}. "
        f"O resultado final documentado foi {Z_MAP[data_point['Z']]}."
    )
    return {
        "text": text,
        "type": "interventional",
        **data_point
    }

def generate_corpus(num_samples: int, interventional_ratio: float, output_path: str):
    """
    Gera um corpus de texto e o salva como um arquivo CSV.

    Args:
        num_samples: O número total de sentenças a serem geradas.
        interventional_ratio: A proporção de sentenças que devem ser do tipo "intervencional".
        output_path: O caminho para salvar o arquivo CSV de saída.
    """
    print(f"Gerando corpus com {num_samples} amostras (intervencional_ratio={interventional_ratio}) ...")
    
    # 1. Gera todos os dados brutos de uma vez de forma vetorizada
    df_raw = sample(num_samples=num_samples)
    
    # 2. Formata as linhas para texto
    num_interventional = int(num_samples * interventional_ratio)
    corpus_list: List[FormattedEntry] = []
    
    for i, row in df_raw.iterrows():
        data_point: PatientData = row.to_dict()
        if i < num_interventional:  
            # Formata as primeiras N linhas como intervencionais
            corpus_list.append(format_interventional(i + 1, data_point))
        else:
            # Formata o restante como observacional
            corpus_list.append(format_observational(i + 1, data_point))
        
    # 3. Cria o DataFrame final e embaralha
    df = pd.DataFrame(corpus_list)
    # Embaralhar as linhas para misturar os tipos de dados
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Garante que o diretório de saída exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Corpus gerado e salvo com sucesso em '{output_path}'.")
    print(f"Total de linhas: {len(df)}")
    print(f"   Observacionais: {len(df[df['type'] == 'observational'])}")
    print(f"   Intervencionais: {len(df[df['type'] == 'interventional'])}")


if __name__ == '__main__':
    # Configuração para executar o script a partir da linha de comando
    parser = argparse.ArgumentParser(
        description="Gera um corpus de texto a partir do modelo causal do micro-mundo."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Número total de sentenças a serem geradas."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Proporção de dados intervencionais (ex: 0.2 para 20%)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/corpus.csv",
        help="Caminho do arquivo CSV de saída."
    )
    
    args = parser.parse_args()
    
    generate_corpus(
        num_samples=args.num_samples,
        interventional_ratio=args.ratio,
        output_path=args.output
    )