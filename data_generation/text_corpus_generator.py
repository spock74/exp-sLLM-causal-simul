# causal_swarm_poc/data_generation/text_corpus_generator.py

import sys
import os
import argparse
import pandas as pd
from typing import Dict, List, Any

# Adiciona o diretório raiz ao path para que possamos importar de 'micro_world'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from micro_world.causal_graph import sample

# --- Templates de Texto ---

# Mapeamentos de valores binários para texto descritivo
C_MAP = {0: "nenhuma condição pré-existente notável", 1: "uma condição pré-existente relevante"}
X_MAP = {0: "placebo", 1: "droga experimental"}
Y_MAP = {0: "nível normal de biomarcador", 1: "nível elevado de biomarcador"}
Z_MAP = {0: "uma recuperação completa", 1: "não apresentou recuperação"}

def format_observational(patient_id: int, data_point: Dict[str, int]) -> Dict[str, Any]:
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

def format_interventional(trial_id: int, data_point: Dict[str, int]) -> Dict[str, Any]:
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

def generate_corpus(num_samples: int, interventional_ratio: float, output_path: str) -> None:
    """
    Gera um corpus de texto e o salva como um arquivo CSV.

    Args:
        num_samples: O número total de sentenças a serem geradas.
        interventional_ratio: A proporção de sentenças que devem ser do tipo "intervencional".
        output_path: O caminho para salvar o arquivo CSV de saída.
    """
    print(f"Gerando corpus com {num_samples} amostras...")
    
    corpus_data: List[Dict[str, Any]] = []
    num_interventional = int(num_samples * interventional_ratio)
    
    # Gerar dados intervencionais
    for i in range(num_interventional):
        data_point = sample(1).to_dict('records')[0]
        formatted_entry = format_interventional(i + 1, data_point)
        corpus_data.append(formatted_entry)
        
    # Gerar dados observacionais
    for i in range(num_samples - num_interventional):
        data_point = sample(1).to_dict('records')[0]
        formatted_entry = format_observational(i + 1, data_point)
        corpus_data.append(formatted_entry)
        
    # Criar um DataFrame e salvar como CSV
    df = pd.DataFrame(corpus_data)
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