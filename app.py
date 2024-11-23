from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from server.instance import server
from src.crontollers.pessoas import *

server.run()

# Inicializar o app Flask
app = Flask(__name__)

# Carregar o modelo treinado e as colunas usadas durante o treinamento
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model, model_columns = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("O arquivo 'diabetes_model.pkl' não foi encontrado. Certifique-se de que ele está no diretório correto.")
except Exception as e:
    raise Exception(f"Erro ao carregar o modelo: {str(e)}")

# Função para processar os dados de entrada
def preprocess_input(data):
    # Criar um DataFrame com as colunas esperadas
    df = pd.DataFrame(data, index=[0])

    # Converter colunas categóricas em variáveis dummy
    df = pd.get_dummies(df, drop_first=True)

    # Adicionar colunas ausentes com valor 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Garantir que todas as colunas esperadas estejam presentes e na ordem correta
    df = df[model_columns]

    return df

# Endpoint de Boas-vindas
@app.route('/')
def home():
    return jsonify({"message": "Bem-vindo à API de Previsão de Diabetes!"})

# Endpoint para prever a presença de diabetes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados enviados via JSON
        input_data = request.json

        # Validar os dados recebidos
        required_fields = [
            'gender', 'age', 'hypertension', 'heart_disease', 
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({"error": f"Campos ausentes: {', '.join(missing_fields)}"}), 400

        # Processar os dados para o formato esperado pelo modelo
        processed_data = preprocess_input(input_data)

        # Realizar a previsão
        prediction = model.predict(processed_data)[0]
        prediction_text = "É provável que haja diabetes." if prediction == 1 else "É improvável haver diabetes."

        # Retornar a previsão como JSON
        return jsonify({
            "prediction": prediction_text,
            "raw_prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro: {str(e)}"}), 500

# Iniciar o servidor
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Define a porta, com valor padrão 5000
    app.run(host="0.0.0.0", port=port, debug=True)
