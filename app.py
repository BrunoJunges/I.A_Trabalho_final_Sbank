# --- SBank Fraud Sentinel: API do Modelo de IA (VERSÃO FINAL COM MÉTRICAS E CORREÇÃO DE BUG) ---

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")

def treinar_e_avaliar_modelo_sbank():
    print("="*60)
    print("Iniciando treinamento e avaliação do modelo SBank (v-final.fix)...")
    print("="*60)
    np.random.seed(42)
    n_samples = 20000

    user_profiles = pd.DataFrame({
        'idade': np.random.normal(loc=32, scale=10, size=n_samples).astype(int).clip(18, 70),
        'renda_mensal': np.random.lognormal(mean=np.log(4500), sigma=0.5, size=n_samples).round(2),
        'score_credito': np.random.randint(300, 950, size=n_samples)
    })
    
    transactions = pd.DataFrame({
        'valor': np.random.lognormal(mean=np.log(68), sigma=0.8, size=n_samples).round(2),
        'hora_do_dia': np.random.randint(0, 24, size=n_samples),
        'eh_internacional': np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    })

    df = pd.concat([user_profiles, transactions], axis=1)
    
    prob_fraude = pd.Series(np.zeros(n_samples), index=df.index)
    prob_fraude += (df['valor'] / 500) * 0.1
    prob_fraude.loc[df['hora_do_dia'] < 6] += 0.25
    prob_fraude.loc[df['score_credito'] < 450] += 0.20
    prob_fraude.loc[df['valor'] > (df['renda_mensal'] * 0.5)] += 0.40
    prob_fraude.loc[df['valor'] > 2500] += 0.3
    
    # --- TRECHO CORRIGIDO ---
    # Em vez de um limiar aleatório, garantimos um percentual fixo de fraudes
    # para evitar o bug de classe com apenas 1 membro.
    
    # 1. Encontrar o valor do limiar que separa os 3% de maiores riscos
    limiar_fraude = prob_fraude.quantile(0.97) 
    
    # 2. Aplicar o limiar para criar a coluna 'eh_fraude'
    df['eh_fraude'] = (prob_fraude > limiar_fraude).astype(int)
    # --- FIM DO TRECHO CORRIGIDO ---

    FEATURES = ['idade', 'renda_mensal', 'score_credito', 'valor', 'hora_do_dia', 'eh_internacional']
    X = df[FEATURES]
    y = df['eh_fraude']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- MÉTRICAS DE PERFORMANCE DO MODELO ---\n")
    print(classification_report(y_test, y_pred, target_names=['Nao Fraude (0)', 'Fraude (1)']))
    print("="*60)
    print("Métricas geradas. O modelo está pronto para uso na API.")
    print("="*60)
    
    model.fit(X, y)
    
    return model, FEATURES

modelo_fraude, FEATURES_ORDENADAS = treinar_e_avaliar_modelo_sbank()

app = Flask(__name__)
CORS(app)

def gerar_justificativa_sbank(transacao):
    razoes = []
    dados = transacao.iloc[0]
    if dados['valor'] > 2500: razoes.append(f"Valor excede o limite médio do cartão SBank (R$2.500)")
    if dados['valor'] > (dados['renda_mensal'] * 0.5): razoes.append("Valor alto em relação à renda mensal")
    if dados['hora_do_dia'] < 6: razoes.append("Horário atípico (madrugada)")
    if dados['score_credito'] < 450 and dados['valor'] > 500: razoes.append("Alto valor para cliente com score baixo")
    if not razoes: return "Nenhum fator de risco óbvio. Análise baseada no padrão geral."
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
        return jsonify({'probabilidade_fraude': round(float(probabilidade_fraude), 4), 'justificativa': justificativa})
    except Exception as e:
        return jsonify({'erro': f'Erro interno no servidor: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
