# causal_swarm_poc/models/causal_reasoner.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel
from typing import cast, Tuple
import numpy as np

# Usaremos a mesma arquitetura de modelo base para consistência.
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/causal_reasoner_finetuned"

def create_reasoner_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Carrega o modelo base e o tokenizador para o Raciocinador Causal.
    A configuração é idêntica à do Extrator de Fatos, pois aplicaremos
    um fine-tuning LoRA separado neste modelo.
    """
    # A implementação desta função é idêntica a create_model_and_tokenizer
    # em fact_extractor.py. Para evitar duplicação, poderíamos movê-la para utils/helpers.py,
    # mas para clareza da PoC, vamos mantê-la aqui por enquanto.
    
    base_model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto"
    ))
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    ))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return base_model, tokenizer

def format_reasoner_prompt(question: str, dag_belief: np.ndarray) -> str:
    """
    Cria o prompt de entrada para o LLM2.
    Combina a pergunta do usuário com a crença causal atual do sistema.
    
    Args:
        question: A pergunta em linguagem natural (ex: "Qual o efeito do tratamento?").
        dag_belief: O vetor binário {0, 1} do estado atual da Rede de Hopfield.
    """
    # Nomes das arestas para tornar o prompt mais legível para a IA
    edge_names = ["C->X", "C->Y", "C->Z", "X->Y", "X->Z", "Y->Z"]
    
    # Constrói uma representação textual do DAG
    active_edges = [name for name, exists in zip(edge_names, dag_belief) if exists == 1]
    dag_representation = f"Hipótese Causal Atual: O grafo contém as seguintes arestas: {', '.join(active_edges)}."

    # Formata o prompt completo usando o template do Phi-3
    prompt = (
        f"<|user|>\n"
        f"Você é um assistente de raciocínio causal. Responda à pergunta do usuário "
        f"considerando estritamente a estrutura causal fornecida. Não use conhecimento externo.\n\n"
        f"--- CONTEXTO CAUSAL ---\n"
        f"{dag_representation}\n\n"
        f"--- PERGUNTA ---\n"
        f"{question}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt

def reason(question: str, dag_belief: np.ndarray, model: PeftModel, tokenizer: PreTrainedTokenizer) -> str:
    """
    Executa uma inferência com o Raciocinador Causal.
    """
    prompt = format_reasoner_prompt(question, dag_belief)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        # Extrai apenas a parte da resposta do assistente
        answer = response_text.split("<|assistant|>")[1].strip()
        return answer
    except IndexError:
        return "Erro: Não foi possível extrair a resposta do modelo."


if __name__ == '__main__':
    print("--- Testando a Estrutura do Módulo Raciocinador Causal (LLM2) ---")
    
    # Este teste apenas demonstra a formatação do prompt.
    # O modelo real precisa ser treinado no Módulo 4.
    
    # Pergunta de exemplo
    test_question = "Se aplicarmos o tratamento em um paciente, qual a probabilidade de ele se recuperar?"
    
    # Crença causal vinda da nossa Rede de Hopfield (o DAG verdadeiro)
    true_dag = np.array([1, 0, 1, 1, 0, 1])
    
    # Como o prompt será montado
    prompt_to_llm2 = format_reasoner_prompt(test_question, true_dag)
    
    print("\n--- Exemplo de Prompt para o LLM2 ---")
    print(prompt_to_llm2)

    # Simulação de uma resposta (o que esperamos que o modelo APRENDA a dizer)
    print("\n--- Exemplo de Resposta Esperada (após o treinamento) ---")
    expected_answer = (
        "Considerando a hipótese causal C->X, C->Z, X->Y, Y->Z, o tratamento (X) não afeta "
        "diretamente o resultado (Z). Seu efeito é mediado pelo biomarcador (Y). "
        "A aplicação do tratamento aumenta a probabilidade de normalizar o biomarcador, "
        "o que por sua vez aumenta significativamente a probabilidade de recuperação."
    )
    print(expected_answer)