# --- SBank Fraud Sentinel: API do Modelo de IA (VERSÃO FINAL - DADOS SBank) ---
# Dependências: Flask, pandas, scikit-learn, lightgbm, Flask-Cors
# Para instalar: pip install Flask pandas scikit-learn lightgbm Flask-Cors

import pandas as pd
import numpy as np
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")

# --- 1. Geração de Dados e Treinamento Baseado no Perfil SBank ---
def treinar_modelo_sbank():
    print("Iniciando treinamento do modelo com perfil de dados SBank (v3)...")
    np.random.seed(42)
    n_samples = 20000

    # Criando perfis de clientes baseados nos dados do SBank
    user_profiles = pd.DataFrame({
        'idade': np.random.normal(loc=32, scale=10, size=n_samples).astype(int).clip(18, 70),
        'renda_mensal': np.random.lognormal(mean=np.log(4500), sigma=0.5, size=n_samples).round(2),
        'score_credito': np.random.randint(300, 950, size=n_samples)
    })
    
    # Criando transações baseadas nos dados do SBank (ticket médio ~R$68)
    transactions = pd.DataFrame({
        'valor': np.random.lognormal(mean=np.log(68), sigma=0.8, size=n_samples).round(2),
        'hora_do_dia': np.random.randint(0, 24, size=n_samples),
        'eh_internacional': np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    })

    df = pd.concat([user_profiles, transactions], axis=1)
    
    # Lógica de fraude mais sofisticada usando o perfil do cliente
    prob_fraude = pd.Series(np.zeros(n_samples), index=df.index)
    prob_fraude += (df['valor'] / 500) * 0.1 # Risco por valor, normalizado
    prob_fraude.loc[df['hora_do_dia'] < 6] += 0.25
    prob_fraude.loc[df['score_credito'] < 450] += 0.20
    prob_fraude.loc[df['valor'] > (df['renda_mensal'] * 0.5)] += 0.40 # <-- Lógica de negócio chave!
    prob_fraude.loc[df['valor'] > 2500] += 0.3 # <-- Limite médio do cartão SBank
    
    df['eh_fraude'] = (prob_fraude > np.random.uniform(0.5, 0.95, n_samples)).astype(int)

    print(f"Dataset de SBank criado com {n_samples} amostras.")
    print(f"Distribuição de fraude:\n{df['eh_fraude'].value_counts(normalize=True) * 100}")

    FEATURES = ['idade', 'renda_mensal', 'score_credito', 'valor', 'hora_do_dia', 'eh_internacional']
    X = df[FEATURES]
    y = df['eh_fraude']
    
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X, y)
    
    print("Modelo SBank treinado com sucesso!")
    return model, FEATURES

modelo_fraude, FEATURES_ORDENADAS = treinar_modelo_sbank()

# --- 2. API Flask com Justificativas Aprimoradas ---
app = Flask(__name__)
CORS(app)

def gerar_justificativa_sbank(transacao):
    """Gera justificativas baseadas no perfil do cliente SBank."""
    razoes = []
    dados = transacao.iloc[0]
    if dados['valor'] > 2500:
        razoes.append(f"Valor excede o limite médio do cartão SBank (R$2.500)")
    if dados['valor'] > (dados['renda_mensal'] * 0.5):
        razoes.append("Valor da transação é alto em relação à renda mensal do cliente")
    if dados['hora_do_dia'] < 6:
        razoes.append("Horário atípico (madrugada)")
    if dados['score_credito'] < 450 and dados['valor'] > 500:
        razoes.append("Alto valor para um cliente com score de crédito baixo")
    if not razoes:
        return "Nenhum fator de risco óbvio. Análise baseada no padrão geral do modelo."
    return " | ".join(razoes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados = request.get_json()
        if not dados or any(key not in dados for key in FEATURES_ORDENADAS):
            return jsonify({'erro': f'Chaves obrigatórias ausentes: {FEATURES_ORDENADAS}'}), 400

        transacao = pd.DataFrame([dados])[FEATURES_ORDENADAS]
        probabilidade_fraude = modelo_fraude.predict_proba(transacao)[:, 1][0]
        justificativa = gerar_justificativa_sbank(transacao)

        return jsonify({
            'probabilidade_fraude': round(float(probabilidade_fraude), 4),
            'justificativa': justificativa
        })
    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
