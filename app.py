from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla, new_obj_ic_jack_pris_html
from theory_texts import texto_01
from metapy_toolbox import initial_population_01, genetic_algorithm_01
import io
from io import BytesIO
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Criar um widget para exibir logs no Streamlit
class StreamlitLogger:
    def __init__(self):
        self.logs = ""

    def write(self, message):
        if message.strip():
            self.logs += message + "\n"
            st.session_state.logs = self.logs

    def flush(self):
        pass

if "logs" not in st.session_state:
    st.session_state.logs = ""

log_area = StreamlitLogger()


def ag_monte_carlo(g_ext: float, q: float, l: float, f_c: float, f_cj: float, phi_a: float, 
                   phi_b: float, psi: float, perda_inicial: float, perda_final: float, iterations: int, 
                   pop_size: int, pres_min: float, pres_max: float, exc_min: float, exc_max: float, 
                   width_min: float, width_max: float, height_min: float, height_max: float) -> tuple[pd.DataFrame, plt.Figure]:
    
    # Configuração do logger para capturar logs em tempo real
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # Placeholder para logs e barra de progresso
    log_area = st.empty()
    progress_bar = st.progress(0)
    logger.info(f"{texts["logger_start"]}")
    
    # Configuração de parâmetros para processamento monte carlo
    n_lambda = 10      
    n_length = 5000    
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = n_length
    np.random.seed(42)
    p_samples = np.random.uniform(p[0], p[1], n)
    e_p_samples = np.random.uniform(e_p[0], e_p[1], n)
    bw_samples = np.random.uniform(bw[0], bw[1], n)
    h_samples = np.random.uniform(h[0], h[1], n)

    # Criação do dataframe
    df = pd.DataFrame({'pk (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples})
    a_c_list, r_list, rig_list, g_lists = [], [], [], []
    logger.info(f"{texts["logger_1"]}")
    # Definir o intervalo para atualização
    update_interval = 100  # Atualiza o progress bar a cada 100 iterações

    # Iteração para avaliação de cada amostra
    for i, row in df.iterrows():
        fixed_variables = {
            'g (kN/m)': g_ext, 'q (kN/m)': q, 'l (m)': l, 'tipo de seção': 'retangular',
            'fck,ato (kPa)': f_cj * 1E3, 'fck (kPa)': f_c * 1E3, 'fator de fluência para o ato': phi_a,
            'fator de fluência para o serviço': phi_b, 'flecha limite de fabrica (m)': l/1000,
            'flecha limite de serviço (m)': l/250, 'coeficiente parcial para carga q': psi,
            'perda inicial de protensão (%)': perda_inicial, 'perda total de protensão (%)': perda_final
        }
        of, g = new_obj_ic_jack_priscilla([row['pk (kN)'], row['e_p (m)'], row['bw (m)'], row['h (m)']], fixed_variables)
        a_c_list.append(of[0])
        r_list.append(of[1])
        g_lists.append(g)
        # Atualiza logs em tempo real a cada N iterações
        if i % update_interval == 0:
            log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_sample_{i}")
            progress_bar.progress((i + 1) / n_length)
    # Atualiza uma última vez após o processamento
    log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_sample_final")

    # Criação das colunas de restrições e função objetivo
    df['a_c (m²)'] = a_c_list
    df['r (%)'] = r_list
    for idx, g_list in enumerate(zip(*g_lists)):
        df[f'g_{idx}'] = g_list

    df = df[(df[[col for col in df.columns if col.startswith('g_')]] <= 0).all(axis=1)].reset_index(drop=True)
    ac_min, ac_max = df['a_c (m²)'].min(), df['a_c (m²)'].max()
    lambda_list = np.linspace(ac_min, ac_max, n_lambda)
    results = []

    # Montando a fronteira eficiente
    logger.info(f"{texts["logger_2"]}")
    for iter_var, lambda_value in enumerate(lambda_list):
        logger.info(f"{texts["logger_3"]} {(iter_var+1)*10} % ...")
        variaveis_proj = {
            'g (kN/m)': g_ext, 'q (kN/m)': q, 'l (m)': l, 'tipo de seção': 'retangular',
            'fck,ato (kPa)': f_cj * 1E3, 'fck (kPa)': f_c * 1E3, 'lambda': lambda_value, 'rp': 1E6,
            'fator de fluência para o ato': phi_a, 'fator de fluência para o serviço': phi_b,
            'flecha limite de fabrica (m)': l/1000, 'flecha limite de serviço (m)': l/250,
            'coeficiente parcial para carga q': psi, 'perda inicial de protensão (%)': perda_inicial,
            'perda total de protensão (%)': perda_final
        }
        algorithm_setup = {
                            'number of iterations': int(iterations),
                            'number of population': int(pop_size),
                            'number of dimensions': 4,
                            'x pop lower limit': [pres_min, exc_min, width_min, height_min],
                            'x pop upper limit': [pres_max, exc_max, width_max, height_max],
                            'none variable': variaveis_proj,
                            'objective function': obj_ic_jack_priscilla,
                            'algorithm parameters': {
                                'selection': {'type': 'roulette'},
                                'crossover': {'crossover rate (%)': 90, 'type': 'linear'},
                                'mutation': {'mutation rate (%)': 20, 'type': 'hill climbing', 'cov (%)': 15, 'pdf': 'uniform'},
                            }
                        }

        # algorithm_setup = {   
        #                     'number of iterations': int(iterations),
        #                     'number of population': int(pop_size),
        #                     'number of dimensions': 4,
        #                     'x pop lower limit': [pres_min, exc_min, width_min, height_min],
        #                     'x pop upper limit': [pres_max, exc_max, width_max, height_max],
        #                     'none variable': variaveis_proj,
        #                     'objective function': obj_ic_jack_priscilla,
        #                     'algorithm parameters': {
        #                                                 'attractiveness': {'gamma': 'auto', 'beta_0': 0.98},
        #                                                 'female population': {'number of females': 10},
        #                                                 'mutation': {
        #                                                                 'mutation rate (%)': 100,
        #                                                                 'type': 'chaotic map 01',
        #                                                                 'alpha': 4,
        #                                                                 'number of tries': 5,
        #                                                             }
        #                                             }
        #                     }

        of_best = []
        df_resume_best = []
        for _ in range(10):
            init_pop = initial_population_01(algorithm_setup['number of population'],
                                    algorithm_setup['number of dimensions'],
                                    algorithm_setup['x pop lower limit'],
                                    algorithm_setup['x pop upper limit'])
            settings = [algorithm_setup, init_pop, None]
            _, df_resume, _, _ = genetic_algorithm_01(settings)
            #_, df_resume, _, _ = gender_firefly_01(settings)
            df_resume_best.append(df_resume)
            of_best.append(df_resume.iloc[-1]['OF BEST'])
        status = of_best.index(min(of_best))
        best_result_row = df_resume_best[status].iloc[-1]
        of, g = new_obj_ic_jack_priscilla([best_result_row['X_0_BEST'],
                                           best_result_row['X_1_BEST'],
                                           best_result_row['X_2_BEST'],
                                           best_result_row['X_3_BEST']], variaveis_proj)
        result = {
            'pk (kN)': best_result_row['X_0_BEST'],  
            'ep (m)': best_result_row['X_1_BEST'],  
            'bw (m)': best_result_row['X_2_BEST'],  
            'h (m)': best_result_row['X_3_BEST'],  
            'a_c (m²)': of[0],  
            'r (%)': of[1] 
        }
        for i, g_value in enumerate(g):
            result[f'G_{i}'] = g_value
        results.append(result)

        # Atualiza logs
        log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_{iter_var}")
        progress_bar.progress((iter_var + 1) / n_lambda)

    logger.info(f"{texts["logger_4"]}")
    logger.info(f"{texts["logger_end"]}")
    log_area.text_area("Logs", log_buffer.getvalue(), height=250, key="log_area_final")


    df_results = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # Gráfico de dispersão
    ax1.scatter(df_results['a_c (m²)'], df_results['r (%)'], color='red', label=f'{texts["graph_label_1"]}') # AG
    df_results = df_results.sort_values(by='a_c (m²)')
    ax1.plot(df_results['a_c (m²)'], df_results['r (%)'], color='red', linestyle='-', linewidth=2)

    ax1.scatter(df['a_c (m²)'], df['r (%)'], color='#dcdcdc', label='Monte Carlo')                     # Monte Carlo
    ax1.title.set_text(f'{texts["graph_label_1_title"]}')
    ax1.set_xlabel(f'{texts["graph_x"]}', fontsize=14)
    ax1.set_ylabel(f'{texts["graph_y"]}', fontsize=14)
    ax1.legend(loc='lower left')

    # Gráfico da fronteira eficiente
    ax2.scatter(df_results['a_c (m²)'], df_results['r (%)'], color='red', label=f'{texts["graph_label_1"]}') # AG
    df_results = df_results.sort_values(by='a_c (m²)')
    ax2.plot(df_results['a_c (m²)'], df_results['r (%)'], color='red', linestyle='-', linewidth=2)
    ax2.title.set_text(f'{texts["graph_label_2_title"]}')
    ax2.set_xlabel(f'{texts["graph_x"]}', fontsize=14)
    ax2.set_ylabel(f'{texts["graph_y"]}', fontsize=14)

    return df_results, fig


def generate_html_download(pres_min: float, pres_max: float, exc_min: float, exc_max: float, width_min: float, width_max: float, height_min: float, height_max: float, g_ext: float, q: float, l: float, f_c: float, f_cj: float, phi_a: float, phi_b: float, psi: float, perda_inicial: float, perda_final: float):
    # Configuração de parâmetros para processamento monte carlo    
    n_length = 5000    
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = n_length
    np.random.seed(42)
    p_samples = np.random.uniform(p[0], p[1], n)
    e_p_samples = np.random.uniform(e_p[0], e_p[1], n)
    bw_samples = np.random.uniform(bw[0], bw[1], n)
    h_samples = np.random.uniform(h[0], h[1], n)

    # Criação do dataframe
    df = pd.DataFrame({'pk (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples})
    print(len(df))

    # Iteração para avaliação de cada amostra
    for i, row in df.iterrows():
        fixed_variables = {
                    'g (kN/m)': g_ext, 'q (kN/m)': q, 'l (m)': l, 'tipo de seção': 'retangular',
                    'fck,ato (kPa)': f_cj * 1E3, 'fck (kPa)': f_c * 1E3, 'fator de fluência para o ato': phi_a,
                    'fator de fluência para o serviço': phi_b, 'flecha limite de fabrica (m)': l/1000,
                    'flecha limite de serviço (m)': l/250, 'coeficiente parcial para carga q': psi,
                    'perda inicial de protensão (%)': perda_inicial, 'perda total de protensão (%)': perda_final
                }
        html_content = new_obj_ic_jack_pris_html([df['pk (kN)'], df['e_p (m)'], df['bw (m)'], df['h (m)']], fixed_variables)

    return html_content



st.write("""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: justify;
        }
        h2 {
            text-align: justify;
        }
    </style>
</head>
        """, unsafe_allow_html=True)

# Carregar traduções do JSON
with open("translations.json", "r", encoding="utf-8") as file:
    translations = json.load(file)

if "lang" not in st.session_state:
    st.session_state.lang = "pt" 

col1, col2 = st.columns(2)
with col1:
    if st.button(translations["pt"]["button_pt"]):  
        st.session_state.lang = "pt"
with col2:
    if st.button(translations["en"]["button_en"]): 
        st.session_state.lang = "en"

texts = translations[st.session_state.lang]

st.title(texts["title"])
st.write(texts["description"])

texto_01()

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
    iterations = st.number_input(texts["iterations"], value=150, step=1)
    pres_min = st.number_input(texts["prestressed_min"], value=None)
    exc_min = st.number_input(texts["eccentricity_min"], value=None)
    width_min = st.number_input(texts["width_min"], value=None)
    height_min = st.number_input(texts["height_min"], value=None)

with col4:
    pop_size = st.number_input(texts["pop_size"], value=25, step=1)
    pres_max = st.number_input(texts["prestressed_max"], value=None)
    exc_max = st.number_input(texts["eccentricity_max"], value=None)
    width_max = st.number_input(texts["width_max"], value=None)
    height_max = st.number_input(texts["height_max"], value=None)

if st.button(texts["run_simulation"]):
    html_content = generate_html_download(pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max, g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final)

    towrite = io.BytesIO()
    towrite.write(html_content.encode("utf-8"))
    towrite.seek(0)
    st.download_button("Baixar Resultados", data=towrite, file_name="resultados.html", mime="text/html")

#     df_results, fig = ag_monte_carlo(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)
#     st.session_state.df_results = df_results
#     st.session_state.fig = fig

# if "df_results" in st.session_state and "fig" in st.session_state:
#     df_results = st.session_state.df_results
#     fig = st.session_state.fig

#     # Exibir o gráfico e os resultados
#     st.subheader(texts["results"])
#     df_results_eng = df_results.copy().map(lambda x: f"{x:.3e}" if isinstance(x, (int, float)) else x)
#     st.write(df_results_eng)
#     st.pyplot(fig)

#     # Criar o arquivo para download
#     towrite_pareto = BytesIO()
#     with pd.ExcelWriter(towrite_pareto, engine="xlsxwriter") as writer:
#         df_results_eng.to_excel(writer, index=False, sheet_name="Pareto Front")
#     towrite_pareto.seek(0)
#     st.download_button(texts["download"], towrite_pareto, f"{texts["xlsx_name"]}.xlsx",
#                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
