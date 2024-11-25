from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os


data = pd.read_csv('diabetes_dataframe.csv') 

app = Flask(__name__)
app.secret_key = "sua_chave_secreta"
show_graphs = False 

# Carregar o modelo treinado e as colunas usadas durante o treinamento
with open('diabetes_model.pkl', 'rb') as file:
    model, model_columns = pickle.load(file)

# Função para converter a entrada do formulário em formato adequado para o modelo
def preprocess_input(data):
    # Criar um DataFrame com as colunas esperadas
    df = pd.DataFrame(data, index=[0])
    
    # Converte as colunas categóricas em variáveis dummy
    df = pd.get_dummies(df, drop_first=True)

    # Adiciona colunas ausentes com valor 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Garantir que todas as colunas esperadas estejam presentes
    df = df[model_columns]

    return df

def formatar_numeros(form_data):
    campos_para_formatar = ['bmi', 'HbA1c_level']
    for campo in campos_para_formatar:
        if campo in form_data:
            form_data[campo] = form_data[campo].replace(",", ".")
            try:
                # Valida se o valor é um número
                form_data[campo] = float(form_data[campo])
            except ValueError:
                raise ValueError(f"Valor inválido para {campo}: {form_data[campo]}")
    return form_data

@app.route('/')
def home():
    global show_graphs
    imc_value = session.get('imc_value', "Seu IMC")
    return render_template('index.html', prediction="", imc_value=imc_value, show_graphs=show_graphs)

@app.route('/toggle_graphs', methods=['POST'])
def toggle_graphs():
    global show_graphs
    show_graphs = not show_graphs
     # Captura a posição do scroll enviada pelo formulário
    scroll_position = request.form.get('scroll_position', '0')

    # Redireciona para a página inicial com a posição do scroll como parâmetro
    return redirect(f"/?scroll_position={scroll_position}")

@app.route('/imc', methods=['GET', 'POST'])
def pag_imc():
    botao_visivel = False
    if request.method == 'POST':
        altura_input = request.form.get('altura', '').replace(",", ".")
        peso_input = request.form.get('peso', '').replace(",", ".")

        try:
            # Conversão de valores
            altura = float(altura_input)
            peso = float(peso_input)

            # Validação dos valores
            if altura <= 0 or peso <= 0:
                return render_template('imc.html', imc_resultado="Erro: Altura e peso devem ser maiores que zero.", botao_visivel=botao_visivel)

            # Cálculo do IMC
            imc = peso / (altura ** 2)
            botao_visivel = True
            session['imc_value'] = f"{imc:.2f}"  # Armazenar IMC na sessão
            
            return render_template('imc.html', imc_resultado=f"{imc:.2f}", botao_visivel=botao_visivel)

        except ValueError:
            return render_template('imc.html', imc_resultado="Erro: Insira valores válidos para peso e altura.", botao_visivel=botao_visivel)
    else:
        return render_template('imc.html', botao_visivel=botao_visivel)
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Coletar dados do formulário
        form_data = request.form.to_dict()
        
        # Formatar os números e validar
        form_data = formatar_numeros(form_data)

        # Estruturar os dados para o modelo
        data = {
            'gender': form_data['gender'],
            'age': int(form_data['age']),
            'hypertension': form_data['hypertension'],
            'heart_disease': form_data['heart_disease'],
            'smoking_history': form_data['smoking_history'],
            'bmi': form_data['bmi'],
            'HbA1c_level': form_data['HbA1c_level'],
            'blood_glucose_level': int(form_data['blood_glucose_level'])
        }
        
        # Convertendo para o formato adequado
        processed_data = preprocess_input(data)
        
        # Realizar a previsão
        prediction = model.predict(processed_data)[0]
        
        if float(prediction) == 0.0:
            resultado_texto = "É improvável que tenha diabetes."
        elif float(prediction) == 1.0:
            resultado_texto = "É provável que tenha diabetes."
        else:
        # Caso ocorra um valor decimal inesperado
            resultado_texto = f"Resultado: {round(float(prediction), 2)}"

        # Captura a posição do scroll enviada pelo formulário
        scroll_position = request.form.get('scroll_position', '0')
        
        # Renderiza a página inicial com o resultado e a posição do scroll
        return redirect(f"/?scroll_position={scroll_position}&prediction={resultado_texto}")

        #return render_template('index.html', prediction=resultado_texto)
    except Exception as e:
        # Log de erro
        return f'Ocorreu um erro: {e}'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Define a porta a partir da variável de ambiente, com 5000 como fallback
    app.run(host="0.0.0.0", port=port)