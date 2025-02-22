from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla
from metapy_toolbox import initial_population_01, genetic_algorithm_01
import io
from io import BytesIO
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from metapy_toolbox import gender_firefly_01
from my_example import my_obj_function
import time

def firefy_gender(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, 
                   iterations, pop_size, pres_min, pres_max, exc_min, exc_max, 
                   width_min, width_max, height_min, height_max):
    
    # # Configuração do logger para capturar logs em tempo real
    # log_buffer = io.StringIO()
    # handler = logging.StreamHandler(log_buffer)
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # logger.addHandler(handler)
    # # Placeholder para logs e barra de progresso
    # log_area = st.empty()
    # progress_bar = st.progress(0)
    # logger.info("Iniciando simulação de Monte Carlo...")
    
    # Configuração de parâmetros para processamento monte carlo
    n_lambda = 10      
    n_length = 500    
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = n_length
    n_reps = 5
    np.random.seed(42)
    p_samples = np.random.uniform(p[0], p[1], n)
    e_p_samples = np.random.uniform(e_p[0], e_p[1], n)
    bw_samples = np.random.uniform(bw[0], bw[1], n)
    h_samples = np.random.uniform(h[0], h[1], n)

    # Criação do dataframe
    df = pd.DataFrame({'p (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples})
    a_c_list, r_list, rig_list, g_lists = [], [], [], []
    # logger.info(f"Processing samples...")
    # # Definir o intervalo para atualização
    # update_interval = 100  # Atualiza o progress bar a cada 100 iterações

    # Iteração para avaliação de cada amostra
    for i, row in df.iterrows():
        fixed_variables = {
            'g (kN/m)': g_ext, 'q (kN/m)': q, 'l (m)': l, 'tipo de seção': 'retangular',
            'fck,ato (kPa)': f_cj * 1E3, 'fck (kPa)': f_c * 1E3, 'fator de fluência para o ato': phi_a,
            'fator de fluência para o serviço': phi_b, 'flecha limite de fabrica (m)': l/1000,
            'flecha limite de serviço (m)': l/250, 'coeficiente parcial para carga q': psi,
            'perda inicial de protensão (%)': perda_inicial, 'perda total de protensão (%)': perda_final
        }
        of, g = new_obj_ic_jack_priscilla([row['p (kN)'], row['e_p (m)'], row['bw (m)'], row['h (m)']], fixed_variables)
        a_c_list.append(of[0])
        r_list.append(of[1])
        g_lists.append(g)
        # # Atualiza logs em tempo real a cada N iterações
        # if i % update_interval == 0:
        #     log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_sample_{i}")
        #     progress_bar.progress((i + 1) / n_length)
    # # Atualiza uma última vez após o processamento
    # log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_sample_final")

    # Criação das colunas de restrições e função objetivo
    df['a_c (m²)'] = a_c_list
    df['r (%)'] = r_list
    for idx, g_list in enumerate(zip(*g_lists)):
        df[f'g_{idx}'] = g_list

    # Retirada do dataframe dos valores que não atendem as restrições
    df = df[(df[[col for col in df.columns if col.startswith('g_')]] <= 0).all(axis=1)].reset_index(drop=True)

    # Prepação do modelo para o algoritmo genético
    ac_min, ac_max = df['a_c (m²)'].min(), df['a_c (m²)'].max()

    # Lista de possíveis áreas para a busca multiobjetivo
    lambda_list = np.linspace(ac_min, ac_max, n_lambda)
    results = []

    # Montando a fronteira eficiente
    for iter_var, lambda_value in enumerate(lambda_list):
        # logger.info(f"Iteration {iter_var + 1}/{n_lambda}.")

        # Atribuição dos valores de entrada do AG
        variaveis_proj = {
            'g (kN/m)': g_ext, 'q (kN/m)': q, 'l (m)': l, 'tipo de seção': 'retangular',
            'fck,ato (kPa)': f_cj * 1E3, 'fck (kPa)': f_c * 1E3, 'lambda': lambda_value, 'rp': 1E6,
            'fator de fluência para o ato': phi_a, 'fator de fluência para o serviço': phi_b,
            'flecha limite de fabrica (m)': l/1000, 'flecha limite de serviço (m)': l/250,
            'coeficiente parcial para carga q': psi, 'perda inicial de protensão (%)': perda_inicial,
            'perda total de protensão (%)': perda_final
        }
            
        # Algorithm setup
        setup = {   
                    'number of iterations': int(iterations),
                    'number of population': int(pop_size),
                    'number of dimensions': 4,
                    'x pop lower limit': [pres_min, exc_min, width_min, height_min],
                    'x pop upper limit': [pres_max, exc_max, width_max, height_max],
                    'none variable': variaveis_proj,
                    'none variable': variaveis_proj,
                    'objective function': obj_ic_jack_priscilla,
                    'algorithm parameters': {
                                                'attractiveness': {'gamma': 'auto', 'beta_0': 0.98},
                                                'female population': {'number of females': 5},
                                                'mutation': {
                                                                'mutation rate (%)': 20,
                                                                'type': 'chaotic map 01',
                                                                'alpha': 4,
                                                                'number of tries': 5,
                                                            }
                                            }
                }

        # Initial guess
        # init_pop = [[-0.74, 1.25],
        #             [3.58, -3.33]]

        # random guess
        from metapy_toolbox import initial_population_01
        init_pop = initial_population_01(setup['number of population'],
                                        setup['number of dimensions'],
                                        setup['x pop lower limit'],
                                        setup['x pop upper limit'])

        # Seed
        seed = None


        settings = [setup, init_pop, seed]
        _, df_resume, _, _ = gender_firefly_01(settings)

        best_result_row = df_resume.iloc[-1]

        # Avaliando as restriçõs do resultado best encontrado
        of, g = new_obj_ic_jack_priscilla([best_result_row['X_0_BEST'], 
                                           best_result_row['X_1_BEST'], 
                                           best_result_row['X_2_BEST'], 
                                           best_result_row['X_3_BEST']], variaveis_proj)
        result = {
            'p (kN)': f"{best_result_row['X_0_BEST']:.3e}",  
            'ep (m)': f"{best_result_row['X_1_BEST']:.3e}",  
            'bw (m)': f"{best_result_row['X_2_BEST']:.3e}",  
            'h (m)': f"{best_result_row['X_3_BEST']:.3e}",  
            'a_c (m²)': f"{of[0]:.3e}",  
            'r (%)': f"{of[1]:.3e}"  
        }

        for i, g_value in enumerate(g):
            result[f'G_{i}'] = f"{g_value:.3e}" 

        results.append(result)

    #     # Atualiza logs
    #     log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_{iter_var}")
    #     progress_bar.progress((iter_var + 1) / n_lambda)

    # logger.info("Finished simulation")

    # Salvando os resultados
    df_results = pd.DataFrame(results)

    # Gerando a figura para tela
    fig, ax = plt.subplots()
    ax.scatter(df_results['a_c (m²)'], df_results['r (%)'], color='red', label='Fronteira eficiente') # AG
    ax.scatter(df['a_c (m²)'], df['r (%)'], color='#dcdcdc', label='Monte Carlo')                     # Monte Carlo
    ax.set_xlabel('Área da seção (m²)', fontsize=14)
    ax.set_ylabel('Carga $g$ estabilizada (%)', fontsize=14)
    ax.legend(loc='lower left')

    # Exibindo os resultados
    st.subheader("Resultados")
    st.write(df_results)
    st.pyplot(fig)



# Carregar traduções do JSON
with open("translations.json", "r", encoding="utf-8") as file:
    translations = json.load(file)

# Inicializar idioma na sessão se não estiver definido
if "lang" not in st.session_state:
    st.session_state.lang = "pt"  # Português como padrão

# Criar um seletor de idioma
col1, col2 = st.columns(2)
with col1:
    if st.button(translations["pt"]["button_pt"]):  # Botão para português
        st.session_state.lang = "pt"
# with col2:
#     if st.button(translations["en"]["button_en"]):  # Botão para inglês
#         st.session_state.lang = "en"

# Obter textos no idioma selecionado
texts = translations[st.session_state.lang]

# Exibir título e descrição
st.title(texts["title"])
st.write(texts["description"])

if __name__ == '__main__':
    st.subheader(texts["parameters"])
    col1, col2 = st.columns(2)

    with col1:
        g_ext = st.number_input(texts["g_ext"], value=None)
        l = st.number_input(texts["l"], value=None)
        f_cj = st.number_input(texts["f_cj"], value=None)
        phi_b = st.number_input(texts["phi_b"], value=None)
        perda_inicial = st.number_input(texts["perda_inicial"], value=None)

    with col2:
        q = st.number_input(texts["q"], value=None)
        f_c = st.number_input(texts["f_c"], value=None)
        phi_a = st.number_input(texts["phi_a"], value=None)
        psi = st.number_input(texts["psi"], value=None)
        perda_final = st.number_input(texts["perda_final"], value=None)

    st.subheader(texts["algorithm_setup"])
    col3, col4 = st.columns(2)

    with col3:
        iterations = st.number_input(texts["iterations"], value=None)
        pres_min = st.number_input(texts["prestressed_min"], value=None)
        exc_min = st.number_input(texts["eccentricity_min"], value=None)
        width_min = st.number_input(texts["width_min"], value=None)
        height_min = st.number_input(texts["height_min"], value=None)

    with col4:
        pop_size = st.number_input(texts["pop_size"], value=None)
        pres_max = st.number_input(texts["prestressed_max"], value=None)
        exc_max = st.number_input(texts["eccentricity_max"], value=None)
        width_max = st.number_input(texts["width_max"], value=None)
        height_max = st.number_input(texts["height_max"], value=None)

    if st.button(texts["run_simulation"]):
        #print(f"Executando simulação com os parâmetros selecionados...")
        inicio = time.time()
        firefy_gender(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)
        fim = time.time()
        print(fim - inicio)
        st.write(fim - inicio)


# # Presstressed beam parameters
# g_ext = 10.75       # Nome no Streamlit: Carga externa permanente (kN/m)
# q = 5.00            # Nome no Streamlit: Carga variável de utilização (kN/m)
# l = 12              # Nome no Streamlit: Vão da viga (m)
# f_c = 30.00         # Nome no Streamlit: Resistência característica à compressão no serviço (MPa) 
# f_cj = 24.56        # Nome no Streamlit: Resistência característica à compressão no ato (MPa)
# phi_a = 2.00        # Nome no Streamlit: Coeficiente de fluência para carregamento no ato
# phi_b = 1.50        # Nome no Streamlit: Coeficiente de fluência para carregamento no serviço
# psi = 0.60          # Nome no Streamlit: Coeficiente ψ redutor para ação variável
# perda_inicial = 5   # Nome no Streamlit: Estimativa pecentual da perda inicial de protensão (%)
# perda_final = 20    # Nome no Streamlit: Estimativa pecentual da perda total de protensão (%)

# # Algorithm parameters
# iterations = 50     # Nome no Streamlit: Número de iterações do otimizador
# pop_size = 30       # Nome no Streamlit: Número de agentes para busca
# n_lambda = 20       # Isso aqui vamos usar sempre 20, o usuário não vai poder editar vai ser "variável de ambiente" nossa  
# n_length = 20000    # Isso aqui vamos usar sempre 20000, o usuário não vai poder editar vai ser "variável de ambiente" nossa  
# pres_min = 100      # Nome no Streamlit: Carga de protensão (kN) - valor inferior
# pres_max = 1000     # Nome no Streamlit: Carga de protensão (kN) - valor superior
# exc_min = 0.10      # Nome no Streamlit: Excentricidade de protensão (m) - valor inferior
# exc_max = 1.00      # Nome no Streamlit: Excentricidade de protensão (m) - valor superior
# width_min = 0.14    # Nome no Streamlit: Largura da seção (m) - valor inferior
# width_max = 2.00    # Nome no Streamlit: Largura da seção (m) - valor superior
# height_min = 0.14   # Nome no Streamlit: Altura da seção (m) - valor inferior
# height_max = 2.00   # Nome no Streamlit: Altura da seção (m) - valor superior