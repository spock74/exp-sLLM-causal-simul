# causal_swarm_poc/models/hopfield_network.py

import numpy as np
from typing import List

class HopfieldNetwork:
    """
    Implementação de uma Rede de Hopfield clássica com atualização síncrona.
    Funciona como uma memória associativa para armazenar e recuperar padrões.
    No nosso caso, os 'padrões' são as estruturas de grafos causais.
    """
    def __init__(self, num_neurons: int):
        """
        Inicializa a rede.
        Args:
            num_neurons: O número de neurônios, que corresponde à dimensionalidade
                         dos nossos vetores de padrão (ex: número de arestas possíveis no DAG).
        """
        self.num_neurons = num_neurons
        # A matriz de pesos, que armazenará a memória da rede.
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns: List[np.ndarray]):
        """
        Treina a rede usando a regra de Hebb para armazenar os padrões fornecidos.
        Os padrões devem ser vetores com valores bipolares {-1, 1}.
        """
        print(f"Treinando a Rede de Hopfield com {len(patterns)} padrões...")
        for p in patterns:
            if p.shape[0] != self.num_neurons:
                raise ValueError("A dimensão do padrão não corresponde ao número de neurônios.")
            self.weights += np.outer(p, p)
        
        # Neurônios não devem ter conexão com eles mesmos.
        np.fill_diagonal(self.weights, 0)
        
        # Normalização (opcional, mas ajuda a estabilizar).
        if len(patterns) > 0:
            self.weights /= len(patterns)
        print("Treinamento concluído.")

    def recall(self, pattern: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        Recupera um padrão a partir de uma entrada (potencialmente ruidosa ou incompleta).
        A rede atualiza seu estado até convergir para um atrator (um padrão memorizado).
        """
        current_state = np.copy(pattern)
        
        for i in range(max_iter):
            # Atualização síncrona: todos os neurônios são atualizados de uma vez.
            new_state = np.sign(np.dot(self.weights, current_state))
            
            # A função np.sign(0) retorna 0. Para manter os estados bipolares,
            # mantemos o valor anterior do neurônio se a entrada for zero.
            new_state[new_state == 0] = current_state[new_state == 0]
            
            if np.array_equal(new_state, current_state):
                return new_state
            
            current_state = new_state
            
        return current_state

    def get_energy(self, pattern: np.ndarray) -> float:
        """Calcula a energia de Lyapunov para um determinado padrão."""
        return -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))

def bipolarize(vector: np.ndarray) -> np.ndarray:
    """Converte um vetor binário {0, 1} para bipolar {-1, 1}."""
    return 2 * vector - 1

def debipolarize(vector: np.ndarray) -> np.ndarray:
    """Converte um vetor bipolar {-1, 1} para binário {0, 1}."""
    return ((vector + 1) / 2).astype(int)


if __name__ == '__main__':
    print("--- Testando o Módulo da Rede de Hopfield ---")

    # Nosso DAG tem 4 nós (C, X, Y, Z). Há 6 arestas direcionadas que estamos considerando
    # como possíveis ou não, para simplificar.
    # Vetor de arestas: [C->X, C->Y, C->Z, X->Y, X->Z, Y->Z]
    NUM_EDGES = 6
    
    # O DAG verdadeiro do nosso micro-mundo, em formato binário {0, 1}
    true_dag_binary = np.array([1, 0, 1, 1, 0, 1])
    
    # Uma hipótese causal incorreta (ex: tratamento afeta diretamente o resultado)
    wrong_dag_binary = np.array([1, 0, 1, 0, 1, 0])

    # Converter para bipolar {-1, 1} para o treinamento da rede
    pattern1 = bipolarize(true_dag_binary)
    pattern2 = bipolarize(wrong_dag_binary)
    
    # Inicializar e treinar a rede para "memorizar" estas duas hipóteses
    network = HopfieldNetwork(num_neurons=NUM_EDGES)
    network.train([pattern1, pattern2])

    print(f"\nPadrão 1 (DAG Verdadeiro) memorizado: {true_dag_binary}")
    print(f"Padrão 2 (DAG Falso) memorizado:      {wrong_dag_binary}")
    
    # Criar um padrão ruidoso para teste. É como o DAG verdadeiro,
    # mas com uma aresta errada (X->Y foi de 1 para 0).
    noisy_input_binary = np.array([1, 0, 1, 0, 0, 1])
    noisy_input = bipolarize(noisy_input_binary)
    
    print(f"\nInput ruidoso (hipótese parcial):    {noisy_input_binary}")
    
    # Pedir à rede para "completar o pensamento"
    recalled_pattern = network.recall(noisy_input)
    recalled_pattern_binary = debipolarize(recalled_pattern)
    
    print(f"Padrão recuperado pela rede:       {recalled_pattern_binary}")
    
    # Verificar o resultado
    if np.array_equal(recalled_pattern_binary, true_dag_binary):
        print("\nSucesso! A rede associou a entrada ruidosa ao DAG verdadeiro.")
    else:
        print("\nFalha. A rede convergiu para um estado diferente.")

    # Comparar as energias para demonstrar o princípio de minimização
    energy_true_dag = network.get_energy(pattern1)
    energy_noisy_input = network.get_energy(noisy_input)
    
    print(f"\nEnergia do atrator (DAG verdadeiro): {energy_true_dag:.4f}")
    print(f"Energia do input ruidoso (instável): {energy_noisy_input:.4f}")