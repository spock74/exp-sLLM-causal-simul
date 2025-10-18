# causal_swarm_poc/models/fact_extractor.py

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import json
from typing import Dict, Tuple, cast, Any

# --- 1. Configuração do Modelo ---

# Usaremos o Phi-3 da Microsoft, que é pequeno, rápido e excelente para fine-tuning.
# Requer a versão mais recente de transformers e accelerate.
# pip install -U transformers accelerate bitsandbytes
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/fact_extractor_finetuned"


def create_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Carrega o modelo e o tokenizador, configurando-os para treinamento com QLoRA.
    """
    print("Carregando modelo base e tokenizador...")

    # SOLUÇÃO APLICADA AQUI:
    # Adicionamos 'attn_implementation="eager"' para compatibilidade com Mac M2.
    base_model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"  # <--- CORREÇÃO CRUCIAL
    ))

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    ))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_wrapped_model = prepare_model_for_kbit_training(base_model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model_for_training = get_peft_model(peft_wrapped_model, lora_config)
    
    print("\nParâmetros treináveis após aplicar LoRA:")
    model_for_training.print_trainable_parameters()
    
    return cast(PreTrainedModel, model_for_training), tokenizer
# --- 2. Preparação dos Dados ---

def create_finetuning_dataset(corpus_path: str) -> Dataset:
    """Carrega o corpus e o formata para o fine-tuning de instrução."""
    df = pd.read_csv(corpus_path)
    
    def format_prompt(row) -> pd.Series:
        # O formato de prompt deve seguir o padrão do modelo específico (neste caso, Phi-3).
        # <|user|>...<|end|><|assistant|>...<|end|>
        
        # A instrução para a IA
        instruction = (
            "Analise o seguinte relatório e extraia os valores para as variáveis "
            "C, X, Y, e Z em formato JSON. C é a condição pré-existente, "
            "X é o tratamento, Y é o biomarcador, e Z é o resultado. "
            "Use 0 para ausente/placebo/normal/recuperado e 1 para presente/droga/elevado/não recuperado."
        )
        
        # O texto de entrada
        input_text = row['text']
        
        # A saída esperada em JSON
        output_json = json.dumps({
            "C": row['C'],
            "X": row['X'],
            "Y": row['Y'],
            "Z": row['Z'],
        })

        # Monta o prompt completo no formato de conversa
        formatted_text = f"<|user|>\n{instruction}\n\nRelatório: \"{input_text}\"<|end|>\n<|assistant|>\n{output_json}<|end|>"
        return pd.Series({"text": formatted_text})

    df_formatted = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df_formatted)
    return dataset

# --- 3. Treinamento ---

def train(corpus_path: str) -> None:
    """Orquestra o processo de fine-tuning do modelo."""
    print("--- Iniciando o Fine-Tuning do Extrator de Fatos (LLM1) ---")
    
    # 1. Carregar modelo e tokenizador
    model, tokenizer = create_model_and_tokenizer()
    
    # 2. Carregar e preparar o dataset
    dataset = create_finetuning_dataset(corpus_path)
    
    # 3. Configurar os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Aumente se tiver mais VRAM
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1, # 1-3 épocas são geralmente suficientes para fine-tuning
        logging_steps=10,
        save_steps=50,
        fp16=True, # Use float16 para acelerar o treinamento
        push_to_hub=False, # Mude para True se quiser salvar no Hub da Hugging Face
    )

    # 4. Iniciar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # O DataCollator agrupa os exemplos em lotes
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Iniciando o treinamento...")
    trainer.train()
    
    print(f"Treinamento concluído. Modelo salvo em '{OUTPUT_DIR}'.")
    trainer.save_model(OUTPUT_DIR)


# --- 4. Inferência (para testar o modelo treinado) ---

def extract_facts(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """Usa o modelo fine-tuned para extrair fatos de um novo texto."""
    instruction = (
        "Analise o seguinte relatório e extraia os valores para as variáveis "
        "C, X, Y, e Z em formato JSON. C é a condição pré-existente, "
        "X é o tratamento, Y é o biomarcador, e Z é o resultado. "
        "Use 0 para ausente/placebo/normal/recuperado e 1 para presente/droga/elevado/não recuperado."
    )
    
    prompt = f"<|user|>\n{instruction}\n\nRelatório: \"{text}\"<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gera a continuação do prompt
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    
    # Decodifica a resposta e extrai apenas a parte gerada pela IA
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_part = response_text.split("<|assistant|>")[1].strip()
    
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        print("Erro: A IA não retornou um JSON válido.")
        print("Saída bruta:", json_part)
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treina ou testa o modelo Extrator de Fatos.")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpus.csv",
        help="Caminho do corpus para treinamento."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa o processo de treinamento."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa um teste de inferência com o modelo treinado."
    )
    
    args = parser.parse_args()
    
    if args.train:
        train(corpus_path=args.corpus)
        
    if args.test:
        print("\n--- Testando Inferência com o Modelo Fine-tuned ---")
        # Carrega o modelo PEFT treinado
        from peft import PeftModel
        base_model, tokenizer = create_model_and_tokenizer()
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        model = model.eval() # Coloca em modo de avaliação

        # Exemplo de texto
        test_text = "Relatório do Paciente 999: O histórico do paciente indica uma condição pré-existente relevante. Foi administrado placebo. A análise subsequente mostrou um nível elevado de biomarcador. O desfecho observado foi não apresentou recuperação."
        
        print(f"\nTexto de entrada:\n{test_text}\n")
        extracted_data = extract_facts(test_text, model, tokenizer)
        print(f"Dados Extraídos (JSON):\n{extracted_data}")
        # Saída esperada: {'C': 1, 'X': 0, 'Y': 1, 'Z': 1}