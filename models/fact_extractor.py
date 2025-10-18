# causal_swarm_poc/models/fact_extractor.py
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer
    
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import argparse
import json
from typing import Dict, Tuple, cast
# --- 1. Configuração do Modelo ---

# Usaremos o Phi-3 da Microsoft, que é pequeno, rápido e excelente para fine-tuning.
# Requer a versão mais recente de transformers e accelerate.
# pip install -U transformers accelerate bitsandbytes einops
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "models/fact_extractor_finetuned"

# causal_swarm_poc/models/fact_extractor.py

# --- Adicione estes imports no topo do seu arquivo ---
from typing import Dict, Tuple, cast
from transformers import PreTrainedModel, PreTrainedTokenizer

# --- Substitua a função existente por esta ---


# --- 2. Preparação dos Dados ---

def create_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Carrega o modelo e o tokenizador, configurando-os para treinamento com QLoRA.
    Retorna uma tupla com o modelo pronto para treinamento e o tokenizador.
    """
    print("Carregando modelo base e tokenizador...")

    # SOLUÇÃO DEFINITIVA: Usamos 'cast' para forçar o tipo correto na origem.
    # Isso quebra a cadeia de incerteza do Pylance.
    base_model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
    ))

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    ))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # A partir daqui, Pylance sabe que base_model é um PreTrainedModel.
    peft_wrapped_model = prepare_model_for_kbit_training(base_model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # E sabe que peft_wrapped_model também é compatível.
    model_for_training = get_peft_model(peft_wrapped_model, lora_config)
    
    print("\nParâmetros treináveis após aplicar LoRA:")
    model_for_training.print_trainable_parameters()
    
    # O cast final aqui pode não ser mais estritamente necessário, mas é
    # uma boa prática para garantir que o tipo de retorno da função seja o correto.
    return cast(PreTrainedModel, model_for_training), tokenizer
# --- 3. Treinamento ---

def train(corpus_path: str) -> None:
    """Orquestra o processo de fine-tuning do modelo."""
    print("--- Iniciando o Fine-Tuning do Extrator de Fatos (LLM1) ---")
    
    model, tokenizer = create_model_and_tokenizer()
    dataset = create_finetuning_dataset(corpus_path)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, # Reduza se tiver OOM (Out of Memory)
        gradient_accumulation_steps=8, # Aumenta o tamanho efetivo do lote
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
        
    print("Iniciando o treinamento...")
    trainer.train()
    
    print(f"Treinamento concluído. Modelo salvo em '{OUTPUT_DIR}'.")
    trainer.save_model(OUTPUT_DIR)


# --- 4. Inferência (para testar o modelo treinado) ---

def extract_facts(text: str, model: PeftModel, tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    """Usa o modelo fine-tuned para extrair fatos de um novo texto."""
    instruction = (
        "Analise o seguinte relatório e extraia os valores para as variáveis "
        "C, X, Y, e Z em formato JSON. C é a condição pré-existente, "
        "X é o tratamento, Y é o biomarcador, e Z é o resultado. "
        "Use 0 para ausente/placebo/normal/recuperado e 1 para presente/droga/elevado/não recuperado."
    )
    
    prompt = f"<|user|>\n{instruction}\n\nRelatório: \"{text}\"<|end|>\n<|assistant|>\n"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        # Encontra a parte da resposta que pertence ao assistente
        json_start_index = response_text.find('{')
        json_str = response_text[json_start_index:]
        data: Dict[str, int] = json.loads(json_str)
        return data
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Erro ao decodificar a saída da IA: {e}")
        print("Saída bruta:", response_text)
        return {}


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
        from peft import PeftModel
        
        base_model: PeftModel
        base_model, tokenizer = create_model_and_tokenizer()
        # Carrega os pesos LoRA adaptados sobre o modelo base
        model: PeftModel = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        model = model.eval()

        test_text = "Relatório do Paciente 999: O histórico do paciente indica uma condição pré-existente relevante. Foi administrado placebo. A análise subsequente mostrou um nível elevado de biomarcador. O desfecho observado foi não apresentou recuperação."
        
        print(f"\nTexto de entrada:\n{test_text}\n")
        extracted_data = extract_facts(test_text, model, tokenizer)
        print(f"Dados Extraídos (JSON):\n{extracted_data}")
        # Saída esperada: {'C': 1, 'X': 0, 'Y': 1, 'Z': 1}