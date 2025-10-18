# **Causal Swarm POC: Um Experimento em Cognição Causal Emergente**

Este repositório contém o código para uma Prova de Conceito (PoC) que explora uma nova arquitetura para embutir raciocínio causal em Modelos de Linguagem Grandes (LLMs). O projeto testa a hipótese de que um ecossistema de LLMs pequenos e especializados, interconectados por redes de memória associativa e guiados por uma função de custo causal-bayesiana, pode aprender uma estrutura causal subjacente a partir de dados textuais, superando a tendência dos LLMs monolíticos de aprenderem apenas correlações superficiais.

## **Hipótese Central**

A conjectura fundamental deste experimento é:

> *Uma inferência causal correta pode emergir como uma mudança de fase em um sistema dinâmico auto-organizado de agentes de linguagem, onde a dinâmica é restringida por um prior causal.*

## **Arquitetura Proposta**

O sistema é composto por quatro componentes principais, projetados para testar a hipótese de forma isolada e controlada:

1.  **Micro-Mundo Causal (`micro_world`):** Um Grafo Acíclico Dirigido (DAG) com 4 variáveis binárias que serve como a "verdade fundamental" (ground truth). Ele modela um cenário médico simplificado com um confundidor, um tratamento, um mediador e um resultado, permitindo a geração de dados e o cálculo de respostas exatas para perguntas causais.

2.  **Gerador de Corpus Textual (`data_generation`):** Um módulo que amostra o micro-mundo para criar um corpus de texto em linguagem natural, simulando relatórios médicos observacionais e dados de ensaios clínicos intervencionais. É este texto, e não os dados brutos, que será fornecido aos modelos de IA.

3.  **O Enxame de IA (`models`):** O núcleo da arquitetura, composto por:
    *   **LLM 1 (Extrator de Fatos):** Um LLM pequeno (2-4B parâmetros) fine-tuned para extrair dados estruturados (JSON) a partir do texto bruto.
    *   **Rede de Hopfield (Espaço de Trabalho Causal):** Uma memória associativa que representa a "crença" atual do sistema sobre a estrutura do DAG. Seus estados de baixa energia (atratores) correspondem a hipóteses causais.
    *   **LLM 2 (Raciocinador Causal):** Outro LLM pequeno que recebe uma pergunta e o estado atual do atrator da Rede de Hopfield como contexto para gerar uma resposta em linguagem natural.

4.  **Função de Custo Híbrida (`training`):** O mecanismo de aprendizado. O sistema é treinado de ponta a ponta com uma função de custo que combina uma perda linguística padrão (cross-entropy) com uma perda causal, definida como a divergência KL entre a distribuição de probabilidade do sistema e a distribuição de probabilidade do modelo causal real.

## **O Experimento**

O objetivo é demonstrar a superioridade do nosso sistema causal em comparação com um sistema de controle puramente correlacional. Ambos os sistemas serão avaliados em sua capacidade de responder a três tipos de perguntas:

1.  **Observacional:** "Qual é a correlação nos dados?" (Ambos devem acertar).
2.  **Intervencional:** "Qual é o efeito causal de uma ação?" (Esperamos que apenas o sistema causal acerte).
3.  **Contrafactual:** "O que teria acontecido se as coisas tivessem sido diferentes?" (O teste definitivo de raciocínio causal).

## **Estrutura do Projeto**

```
causal_swarm_poc/
├── main.py                 # Orquestrador principal do experimento
├── micro_world/            # Módulo 1: Definição do DAG e da "verdade"
├── data_generation/        # Módulo 2: Geração do corpus de texto
├── models/                 # Módulo 3: Implementação dos LLMs e da Rede de Hopfield
├── training/               # Módulo 4: Lógica de treinamento com a função de custo híbrida
├── evaluation/             # Módulo 5: Scripts de avaliação e comparação de modelos
└── utils/                  # Funções de utilidade
```

## **Como Executar**

*(Esta seção será preenchida conforme os módulos são finalizados)*

1.  **Instalar dependências:**
    ```bash
    # Install major packages with Conda:
    conda install -c conda-forge numpy pandas econml

    # Install PyTorch from its official channel:
    conda install pytorch -c pytorch

    # Install remaining dependencies via pip:
    pip install -r requirements.txt
    ```

2.  **Gerar o Corpus de Dados:**
    ```bash
    python data_generation/text_corpus_generator.py --num-samples 20000
    ```

3.  **Treinar os Modelos:**
    ```bash
    python training/train_system.py --model-type causal
    python training/train_system.py --model-type correlational
    ```

4.  **Avaliar os Resultados:**
    ```bash
    python evaluation/evaluate_models.py
    ```


---
### ⚙️ **Ambiente de Desenvolvimento e Solução de Problemas**

A configuração do ambiente para este projeto requer atenção a algumas dependências específicas do ecossistema de IA e configurações do editor de código.

#### **Instalação de Dependências**

É crucial instalar as bibliotecas de Deep Learning na ordem correta e garantir a compatibilidade com sua GPU (se aplicável).

1.  **PyTorch:** A dependência mais importante. Instale a versão correta para sua configuração de hardware (CPU ou versão específica do CUDA) visitando o site oficial: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

2.  **Bibliotecas do Ecossistema Hugging Face:** Após instalar o PyTorch, instale as demais dependências. Recomenda-se usar uma abordagem de instalação limpa para evitar conflitos:

    ```bash
    # Desinstale versões antigas para garantir uma instalação limpa (opcional, mas recomendado)
    pip uninstall datasets transformers accelerate -y

    # Instale as bibliotecas na ordem correta
    pip install datasets
    pip install -U transformers accelerate bitsandbytes peft einops pandas
    ```

#### **Configuração do Editor (VS Code + Pylance)**

Ao trabalhar com o projeto no VS Code, você pode encontrar um aviso `reportMissingTypeStubs` para a biblioteca `datasets`. Isso ocorre porque não há um pacote de "type stubs" oficial para ela. Este aviso não impede a execução do código, mas pode ser desativado para uma experiência de desenvolvimento mais limpa.

1.  Crie um diretório `.vscode` na raiz do projeto, se ele não existir.
2.  Dentro de `.vscode`, crie um arquivo `settings.json`.
3.  Adicione a seguinte configuração para silenciar o aviso:

    ```json
    {
      "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingTypeStubs": "none"
      }
    }
    ```
