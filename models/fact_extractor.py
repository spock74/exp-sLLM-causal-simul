# causal_swarm_poc/models/fact_extractor.py

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer # Mantemos este para o cast, mas o principal é o abaixo
)
# A importação específica sugerida pelo Pylance não é a padrão da comunidade.
# A melhor prática é importar diretamente do pacote principal.
# O aviso do Pylance pode ser um 'falso positivo' ou excessivamente rigoroso.

# A solução mais comum e robusta é manter a importação padrão e,
# se o aviso persistir, suprimir a regra no settings.json,
# pois a comunidade e a documentação oficial usam 'from transformers import AutoTokenizer'.

from peft import LoraConfig, get_peft_model, PeftModel
import argparse
import json
from typing import Dict, Tuple, cast, Any

# --- 1. Configuração Global e do Modelo ---

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/fact_extractor_finetuned"

def create_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Carrega o modelo e o tokenizador otimizados para Apple Silicon (MPS).
    Usa bfloat16 e LoRA com rank reduzido para economizar memória.
    """
    print("Carregando modelo base e tokenizador em bfloat16 para MPS...")

    # Carrega o modelo em bfloat16 para cortar o uso de memória pela metade.
    # Adiciona 'attn_implementation="eager"' para compatibilidade com Mac M2.
    base_model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16
    ))

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    ))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NÃO usamos prepare_model_for_kbit_training aqui, pois não há quantização.
    
    # Configuração LoRA com rank reduzido para otimizar memória.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Aplica o wrapper PEFT diretamente no modelo base.
    model_for_training = get_peft_model(base_model, lora_config)
    
    print("\nParâmetros treináveis após aplicar LoRA (configuração de memória otimizada):")
    model_for_training.print_trainable_parameters()
    
    return cast(PreTrainedModel, model_for_training), tokenizer

# --- 2. Preparação dos Dados ---

def create_finetuning_dataset(corpus_path: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Carrega o corpus e o prepara para o fine-tuning de instrução, incluindo tokenização."""
    df = pd.read_csv(corpus_path)
    
    def format_prompt(row: pd.Series) -> Dict[str, str]:
        instruction = (
            "Analise o seguinte relatório e extraia os valores para as variáveis "
            "C, X, Y, e Z em formato JSON. C é a condição pré-existente, "
            "X é o tratamento, Y é o biomarcador, e Z é o resultado. "
            "Use 0 para ausente/placebo/normal/recuperado e 1 para presente/droga/elevado/não recuperado."
        )
        input_text = row['text']
        output_json = json.dumps({"C": row['C'], "X": row['X'], "Y": row['Y'], "Z": row['Z']})
        formatted_text = f"<|user|>\n{instruction}\n\nRelatório: \"{input_text}\"<|end|>\n<|assistant|>\n{output_json}<|end|>"
        return {"text": formatted_text}

    df_formatted = df.apply(format_prompt, axis=1)
    dataset = Dataset.from_pandas(df_formatted[['text']])
    
    # Adiciona o passo de tokenização
    def tokenize_function(examples):
        max_length = 256
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    final_dataset = tokenized_dataset.map(add_labels, batched=True)
    return final_dataset

# --- 3. Treinamento ---

def train(corpus_path: str) -> None:
    """Orquestra o processo de fine-tuning com otimizações de memória."""
    print("--- Iniciando o Fine-Tuning do Extrator de Fatos (LLM1) ---")
    
    model, tokenizer = create_model_and_tokenizer()
    tokenized_dataset = create_finetuning_dataset(corpus_path, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        gradient_checkpointing=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Iniciando o treinamento...")
    trainer.train()
    
    print(f"Treinamento concluído. Modelo salvo em '{OUTPUT_DIR}'.")
    trainer.save_model(OUTPUT_DIR)

# --- 4. Inferência ---

def extract_facts(text: str, model: PeftModel, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """Usa o modelo fine-tuned para extrair fatos de um novo texto."""
    instruction = (
        "Analise o seguinte relatório e extraia os valores para as variáveis "
        "C, X, Y, e Z em formato JSON. C é a condição pré-existente, "
        "X é o tratamento, Y é o biomarcador, e Z é o resultado. "
        "Use 0 para ausente/placebo/normal/recuperado e 1 para presente/droga/elevado/não recuperado."
    )
    
    prompt = f"<|user|>\n{instruction}\n\nRelatório: \"{text}\"<|end|>\n<|assistant|>\n"
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_part = response_text.split("<|assistant|>")[1].strip()
        return cast(Dict[str, Any], json.loads(json_part))
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Erro ao decodificar a saída da IA: {e}")
        print("Saída bruta:", response_text)
        return {}

# --- 5. Execução Principal ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treina ou testa o modelo Extrator de Fatos.")
    parser.add_argument("--corpus", type=str, default="data/corpus.csv", help="Caminho do corpus para treinamento.")
    parser.add_argument("--train", action="store_true", help="Executa o processo de treinamento.")
    parser.add_argument("--test", action="store_true", help="Executa um teste de inferência com o modelo treinado.")
    
    args = parser.parse_args()
    
    if args.train:
        train(corpus_path=args.corpus)
        
    if args.test:
        print("\n--- Testando Inferência com o Modelo Fine-tuned ---")
        
        # Carrega o modelo base otimizado
        base_model, tokenizer = create_model_and_tokenizer()
        
        # Carrega os pesos LoRA adaptados sobre o modelo base
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        model = model.eval()

        test_text = "Relatório do Paciente 999: O histórico do paciente indica uma condição pré-existente relevante. Foi administrado placebo. A análise subsequente mostrou um nível elevado de biomarcador. O desfecho observado foi não apresentou recuperação."
        
        print(f"\nTexto de entrada:\n{test_text}\n")
        extracted_data = extract_facts(test_text, model, tokenizer)
        print(f"Dados Extraídos (JSON):\n{extracted_data}")
        # Saída esperada: {'C': 1, 'X': 0, 'Y': 1, 'Z': 1}