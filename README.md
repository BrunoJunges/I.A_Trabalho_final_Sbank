# SBank Fraud Sentinel - Sistema de Detecção de Fraude com IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-red.svg)

## 1. Visão Geral

O **SBank Fraud Sentinel** é uma aplicação de demonstração que simula um sistema de detecção de fraudes em tempo real para o banco digital fictício SBank. A solução utiliza um modelo de Machine Learning (IA) para analisar transações e o perfil dos clientes, identificando atividades suspeitas antes que causem prejuízos.

O projeto consiste em duas partes principais:
1.  **Backend (API):** Uma API em Python (Flask) que serve um modelo de IA treinado para calcular a probabilidade de fraude de uma transação.
2.  **Frontend (Dashboard):** Uma interface web em HTML/JS/CSS que simula a tela de um analista de risco, consumindo os dados da API e exibindo alertas em tempo real.

## 2. Problema de Negócio

Como um banco 100% digital, o SBank processa milhões de transações diariamente, o que o torna um alvo para atividades fraudulentas. A detecção tardia de fraudes resulta em:
*   **Perdas financeiras** diretas para o banco e seus clientes.
*   **Queda na confiança** do cliente, afetando a reputação e a retenção.
*   **Custos operacionais** elevados com análises manuais e processos de estorno.

## 3. Solução Proposta

A solução implementa um classificador **LightGBM**, um algoritmo de Machine Learning de alta performance, treinado com dados que simulam a realidade do SBank (baseado nos arquivos CSV fornecidos).

O modelo não analisa apenas a transação de forma isolada, ele enriquece a análise com o **perfil do cliente**, utilizando as seguintes features:
*   **Dados da Transação:** `valor`, `hora_do_dia`, `eh_internacional`.
*   **Dados do Cliente:** `idade`, `renda_mensal`, `score_credito`.

Ao receber os dados de uma nova transação, a API retorna não apenas um **score de fraude** (0 a 1), mas também uma **justificativa textual** que explica os principais fatores de risco (ex: "Valor alto para a renda do cliente", "Horário atípico").

## 4. Funcionalidades da Demonstração

*   **Feed de Transações em Tempo Real:** O dashboard é populado com novas transações a cada poucos segundos.
*   **Análise Preditiva da IA:** Cada transação é enviada à API e recebe um score de fraude.
*   **Alertas Visuais:** Transações com alta probabilidade de fraude são destacadas em vermelho, chamando a atenção do analista.
*   **Visão 360º:** Ao clicar em um alerta, o analista visualiza um painel com os dados completos da transação e do perfil do cliente.
*   **Explicabilidade (XAI):** A justificativa gerada pela IA ajuda o analista a entender o porquê da suspeita.
*   **Interatividade:** O analista pode simular a aprovação ou o bloqueio da transação, e a interface reage a essa ação.

## 5. Tecnologias Utilizadas

*   **Backend:** Python 3, Flask, Flask-CORS, Pandas, Scikit-learn, LightGBM.
*   **Frontend:** HTML5, CSS3, JavaScript (Vanilla).

## 6. Como Executar o Projeto

1.  **Pré-requisitos:** Certifique-se de ter o Python 3.9+ instalado.
2.  **Clone/Baixe o Repositório:** Salve os arquivos `app.py` e `dashboard.html` no mesmo diretório.
3.  **Instale as Dependências:**
    ```bash
    pip install Flask Flask-Cors pandas scikit-learn lightgbm
    ```
4.  **Inicie o Backend:** Execute o servidor Flask no seu terminal.
    ```bash
    python app.py
    ```
5.  **Abra o Frontend:** No seu explorador de arquivos, encontre e clique duas vezes em `dashboard.html`. Ele será aberto no seu navegador padrão.

## 7. Estrutura dos Arquivos

```
sbank-fraud-sentinel/
│
├── app.py             # Backend: API em Flask e Lógica do Modelo de IA
├── dashboard.html     # Frontend: Dashboard do analista de risco
├── passo a passo.txt  # passo a passo para o uso do projeto
└── README.md          # Este arquivo

```
