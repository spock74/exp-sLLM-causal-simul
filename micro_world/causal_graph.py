# causal_swarm_poc/micro_world/causal_graph.py

import numpy as np
import pandas as pd
from typing import Dict, List

# Definindo as equações estruturais do nosso mundo causal como constantes
# para fácil referência e modificação.
P_C1 = 0.5
P_X1_GIVEN_C0 = 0.1
P_X1_GIVEN_C1 = 0.8
P_Y1_GIVEN_X0 = 0.9
P_Y1_GIVEN_X1 = 0.2
P_Z1_GIVEN_Y0_C0 = 0.1
P_Z1_GIVEN_Y0_C1 = 0.4
P_Z1_GIVEN_Y1_C0 = 0.6
P_Z1_GIVEN_Y1_C1 = 0.9

def sample(num_samples: int = 1) -> pd.DataFrame:
    """
    Gera um DataFrame de amostras (pacientes) a partir do modelo causal de forma vetorizada.
    Segue a ordem topológica do DAG: C -> X, C -> Z, X -> Y, Y -> Z.
    """
    # Amostra C para todos os pacientes
    c = np.random.binomial(1, P_C1, size=num_samples)
    
    # Amostra X condicionado a C
    p_x1 = np.where(c == 1, P_X1_GIVEN_C1, P_X1_GIVEN_C0)
    x = np.random.binomial(1, p_x1)
    
    # Amostra Y condicionado a X
    p_y1 = np.where(x == 0, P_Y1_GIVEN_X0, P_Y1_GIVEN_X1)
    y = np.random.binomial(1, p_y1)
    
    # Amostra Z condicionado a Y e C (usando np.select para clareza)
    p_z1 = np.select(
        condlist=[(y==0)&(c==0), (y==0)&(c==1), (y==1)&(c==0), (y==1)&(c==1)],
        choicelist=[P_Z1_GIVEN_Y0_C0, P_Z1_GIVEN_Y0_C1, P_Z1_GIVEN_Y1_C0, P_Z1_GIVEN_Y1_C1]
    )
    z = np.random.binomial(1, p_z1)
    
    return pd.DataFrame({'C': c, 'X': x, 'Y': y, 'Z': z})

def compute_interventional_probability(do_x: int, num_simulations: int = 10000) -> float:
    """
    Calcula P(Z=0 | do(X=x)) usando simulação de Monte Carlo.
    A intervenção "do(X=x)" significa que removemos as setas que entram em X
    e fixamos seu valor.

    Retorna a probabilidade de RECUPERAÇÃO (Z=0).
    """
    # Passo 1: Amostra as variáveis exógenas (C) para todas as simulações de uma vez
    c = np.random.binomial(1, P_C1, size=num_simulations)
    
    # Passo 2: Fixa X com o valor da intervenção
    x = np.full(num_simulations, do_x)
    
    # Passo 3: Propaga o efeito causal para frente (de forma vetorizada)
    p_y1 = np.where(x == 0, P_Y1_GIVEN_X0, P_Y1_GIVEN_X1)
    y = np.random.binomial(1, p_y1)
    
    # Calcula p_z1 usando condições booleanas com np.where
    p_z1 = np.zeros(num_simulations)
    p_z1 = np.where((y == 0) & (c == 0), P_Z1_GIVEN_Y0_C0, p_z1)
    p_z1 = np.where((y == 0) & (c == 1), P_Z1_GIVEN_Y0_C1, p_z1)
    p_z1 = np.where((y == 1) & (c == 0), P_Z1_GIVEN_Y1_C0, p_z1)
    p_z1 = np.where((y == 1) & (c == 1), P_Z1_GIVEN_Y1_C1, p_z1)
    outcomes = np.random.binomial(1, p_z1)
    
    # A probabilidade de recuperação (Z=0) é 1 - a média dos resultados (Z=1)
    prob_recovery: np.floating = 1 - np.mean(outcomes)
    return float(prob_recovery)

def compute_counterfactual(evidence: Dict[str, int], intervention: Dict[str, int], num_simulations: int = 10000) -> float:
    """
    Calcula uma probabilidade contrafactual usando o algoritmo de 3 passos de Pearl.
    Exemplo: "Dado um paciente que tomou a droga (X=1) e se recuperou (Z=0),
             qual seria a probabilidade de recuperação se ele não tivesse tomado a droga?"
             evidence={'X': 1, 'Z': 0}, intervention={'X': 0}

    Retorna a probabilidade de RECUPERAÇÃO (Z=0) no mundo contrafactual.
    """
    # Vetorização do Passo 1: Abduction (Amostragem por Rejeição)
    # Continuamos a gerar amostras até termos 'num_simulations' que correspondam à evidência.
    c_abducted_list: List[np.ndarray] = []
    num_needed = num_simulations
    while num_needed > 0:
        # Gera um lote de candidatos para a variável exógena C
        c_candidate = np.random.binomial(1, P_C1, size=num_needed * 5) # Gera mais para ter chance de encontrar

        # Simula o mundo para frente para cada candidato
        p_x1_candidate = np.where(c_candidate == 1, P_X1_GIVEN_C1, P_X1_GIVEN_C0)
        x_candidate = np.random.binomial(1, p_x1_candidate)

        p_y1_candidate = np.where(x_candidate == 0, P_Y1_GIVEN_X0, P_Y1_GIVEN_X1)
        y_candidate = np.random.binomial(1, p_y1_candidate)

        p_z1_candidate = np.zeros_like(c_candidate, dtype=float)
        p_z1_candidate = np.where((y_candidate == 0) & (c_candidate == 0), P_Z1_GIVEN_Y0_C0, p_z1_candidate)
        p_z1_candidate = np.where((y_candidate == 0) & (c_candidate == 1), P_Z1_GIVEN_Y0_C1, p_z1_candidate)
        p_z1_candidate = np.where((y_candidate == 1) & (c_candidate == 0), P_Z1_GIVEN_Y1_C0, p_z1_candidate)
        p_z1_candidate = np.where((y_candidate == 1) & (c_candidate == 1), P_Z1_GIVEN_Y1_C1, p_z1_candidate)
        z_candidate = np.random.binomial(1, p_z1_candidate)

        # Verifica quais candidatos correspondem à evidência
        evidence_match = np.full(c_candidate.shape, True)
        if 'X' in evidence:
            evidence_match &= (x_candidate == evidence['X'])
        if 'Y' in evidence:
            evidence_match &= (y_candidate == evidence['Y'])
        if 'Z' in evidence:
            evidence_match &= (z_candidate == evidence['Z'])

        # Adiciona os candidatos válidos à nossa lista
        valid_cs = c_candidate[evidence_match]
        c_abducted_list.append(valid_cs)
        num_needed -= len(valid_cs)

    # Concatena e trunca para o número exato de simulações necessárias
    c_abducted = np.concatenate(c_abducted_list)[:num_simulations]

    # Passo 2: Action - Aplicar a intervenção
    x_counterfactual = np.full(num_simulations, intervention['X'])
    
    # Passo 3: Prediction - Calcular o resultado no mundo contrafactual
    p_y1_cf = np.where(x_counterfactual == 0, P_Y1_GIVEN_X0, P_Y1_GIVEN_X1)
    y_cf = np.random.binomial(1, p_y1_cf)

    p_z1_cf = np.zeros(num_simulations, dtype=float)
    p_z1_cf = np.where((y_cf == 0) & (c_abducted == 0), P_Z1_GIVEN_Y0_C0, p_z1_cf)
    p_z1_cf = np.where((y_cf == 0) & (c_abducted == 1), P_Z1_GIVEN_Y0_C1, p_z1_cf)
    p_z1_cf = np.where((y_cf == 1) & (c_abducted == 0), P_Z1_GIVEN_Y1_C0, p_z1_cf)
    p_z1_cf = np.where((y_cf == 1) & (c_abducted == 1), P_Z1_GIVEN_Y1_C1, p_z1_cf)
    z_cf = np.random.binomial(1, p_z1_cf)

    prob_recovery_cf: np.floating = 1 - np.mean(z_cf)
    return float(prob_recovery_cf)

if __name__ == '__main__':
    # Pequeno teste para demonstrar as funções do módulo
    print("--- Testando o Módulo do Micro-Mundo Causal ---")
    
    # 1. Gerar uma amostra de paciente
    paciente_exemplo = sample(1).to_dict('records')[0]
    print(f"\nExemplo de um paciente gerado: {paciente_exemplo}")
    
    # 2. Calcular o Efeito Causal do Tratamento (Intervenção)
    prob_recuperacao_com_droga = compute_interventional_probability(do_x=1)
    prob_recuperacao_com_placebo = compute_interventional_probability(do_x=0)
    
    print(f"\nProb. de recuperação SE TODOS tomassem o tratamento: {prob_recuperacao_com_droga:.2%}")
    print(f"Prob. de recuperação SE NINGUÉM tomasse o tratamento: {prob_recuperacao_com_placebo:.2%}")
    print("Conclusão: O tratamento tem um forte efeito causal positivo.")

    # 3. Calcular um Contrafactual
    # "Para um paciente que tomou a droga (X=1) e NÃO se recuperou (Z=1),
    # qual teria sido a chance de recuperação se ele tivesse recebido o placebo (X=0)?"
    evidencia_paciente = {'X': 1, 'Z': 1}
    intervencao_cf = {'X': 0}
    prob_recuperacao_cf = compute_counterfactual(evidencia_paciente, intervencao_cf)

    print("\nContrafactual: Para um paciente que não se recuperou com a droga,")
    print(f"a chance de recuperação com placebo teria sido: {prob_recuperacao_cf:.2%}")