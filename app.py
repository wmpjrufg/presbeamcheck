from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla
from metapy_toolbox import metaheuristic_optimizer
import io
from io import BytesIO
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

# # Configuração do logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()

# # Criar um widget para exibir logs no Streamlit
# class StreamlitLogger:
#     def __init__(self):
#         self.logs = ""

#     def write(self, message):
#         if message.strip():
#             self.logs += message + "\n"
#             st.session_state.logs = self.logs

#     def flush(self):
#         pass

# if "logs" not in st.session_state:
#     st.session_state.logs = ""

# log_area = StreamlitLogger()


def ag_monte_carlo(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, 
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
    
    # Configuração inicial
    n_lambda = 20      
    n_length = 20000    
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

    df = pd.DataFrame({'p (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples})
    
    a_c_list, r_list, rig_list, g_lists = [], [], [], []

    # logger.info(f"Processing samples...")

    # # Definir o intervalo para atualização
    # update_interval = 100  # Atualiza o progress bar a cada 100 iterações

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

    df['a_c (m²)'] = a_c_list
    df['r'] = r_list

    for idx, g_list in enumerate(zip(*g_lists)):
        df[f'g_{idx}'] = g_list

    df = df[(df[[col for col in df.columns if col.startswith('g_')]] <= 0).all(axis=1)].reset_index(drop=True)

    ac_min, ac_max = df['a_c (m²)'].min(), df['a_c (m²)'].max()

    lambda_list = np.linspace(ac_min, ac_max, n_lambda)
    results = []

    for iter_var, lambda_value in enumerate(lambda_list):
        # logger.info(f"Iteration {iter_var + 1}/{n_lambda}.")

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
                'mutation': {'mutation rate (%)': 20, 'type': 'hill climbing', 'cov (%)': 10, 'pdf': 'gaussian'},
            }
        }

        general_setup = {
            'number of repetitions': 15, 'type code': 'real code',
            'initial pop. seed': [None] * 15, 'algorithm': 'genetic_algorithm_01',
        }

        df_all_reps, df_resume_all_reps, reports, status = metaheuristic_optimizer(algorithm_setup, general_setup)
        best_result_row = df_resume_all_reps[status].iloc[-1]

        of, g = new_obj_ic_jack_priscilla([best_result_row['X_0_BEST'], 
                                           best_result_row['X_1_BEST'], 
                                           best_result_row['X_2_BEST'], 
                                           best_result_row['X_3_BEST']], variaveis_proj)

        result = {
            'lambda': lambda_value,
            'X_0_BEST': best_result_row['X_0_BEST'], 'X_1_BEST': best_result_row['X_1_BEST'],
            'X_2_BEST': best_result_row['X_2_BEST'], 'X_3_BEST': best_result_row['X_3_BEST'],
            'OF_0': of[0], 'OF_1': of[1]
        }

        for i, g_value in enumerate(g):
            result[f'G_{i}'] = g_value

        results.append(result)

    #     # Atualiza logs
    #     log_area.text_area("Logs", log_buffer.getvalue(), height=250, key=f"log_area_{iter_var}")
    #     progress_bar.progress((iter_var + 1) / n_lambda)

    # logger.info("Finished simulation")

    df_results = pd.DataFrame(results)

    # Gerando a figura 
    fig, ax = plt.subplots()
    ax.scatter(df_results['OF_0'], df_results['OF_1'], color='red', label='Fronteira eficiente')
    ax.scatter(df['a_c (m²)'], df['r'], color='#dcdcdc', label='Monte Carlo')

    ax.set_xlabel('Área da seção (m²)', fontsize=14)
    ax.set_ylabel('Carga $g$ estabilizada (%)', fontsize=14)
    ax.legend()

    # Exibindo os resultados
    st.subheader("Resultados")
    st.write(df)
    st.pyplot(fig)

    towrite_pareto = BytesIO()
    with pd.ExcelWriter(towrite_pareto, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Pareto Front")

    towrite_pareto.seek(0)

    st.download_button("Download Fronteira Eficiente", towrite_pareto, "fronteira_eficiente.xlsx", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def monte_carlo(g, q, l, f_c, f_cj, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max):
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = 0 if pop_size == None else int(pop_size)

    if st.button("Run Simulation"):
        np.random.seed(42)
        bw_samples = list(np.random.uniform(bw[0], bw[1], n))
        h_samples = list(np.random.uniform(h[0], h[1], n))
        p_samples = list(np.random.uniform(p[0], p[1], n))
        e_p_samples = list(np.random.uniform(e_p[0], e_p[1], n))
        bw_samples = list(np.random.uniform(bw[0], bw[1], n))
        h_samples = list(np.random.uniform(h[0], h[1], n))
        df = {'p (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples}
        df = pd.DataFrame(df)

        a_c_list = []
        r_list = []
        g_lists = []

        fixed_variables = {
                            'g (kN/m)': g,
                            'q (kN/m)': q,
                            'l (m)': l,
                            'tipo de seção': 'retangular',
                            'tipo de protensão': 'Parcial',
                            'fck,ato (kPa)': f_cj * 1E3,
                            'fck (kPa)': f_c * 1E3,
                            'fator de fluência': 2.5,
                            'flecha limite de fabrica (m)': l/1000,
                            'flecha limite de serviço (m)': l/250,
                            'coeficiente parcial para carga q': 0.60,
                            'perda inicial de protensão (%)': 5,
                            'perda total de protensão (%)': 20
                          }


        for _, row in df.iterrows():
            of, g = new_obj_ic_jack_priscilla([row['p (kN)'], row['e_p (m)'], row['bw (m)'], row['h (m)']], fixed_variables)
            a_c_list.append(of[0])
            r_list.append(of[1])
            g_lists.append(g)
        df['a_c (m²)'] = a_c_list
        df['r'] = r_list

        for idx, g_list in enumerate(zip(*g_lists)):
            df[f'g_{idx}'] = g_list

        df = pd.DataFrame(df)

        df = df[(df[[col for col in df.columns if col.startswith('g_')]] <= 0).all(axis=1)]
        df.reset_index(drop=True, inplace=True)
        # st.subheader("Simulation results")
        # st.table(df) 

        # # Salvando a planilha em um buffer (BytesIO)
        # towrite = BytesIO()
        # with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        #     df.to_excel(writer, index=False, sheet_name="Simulação")
        # towrite.seek(0)  
        # st.download_button(
        #     label="Download results",
        #     data=towrite,
        #     file_name="simulacao_monte_carlo.xlsx",
        #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        # )

        # Plotar gráfico de dispersão e Pareto front
        fix, ax = plt.subplots()
        df_sorted = df.sort_values(by='a_c (m²)', ascending=True).reset_index(drop=True)
        pareto_indices = []
        max_r = -float('inf')
        for idx, row in df_sorted.iterrows():
            if row['r'] > max_r:
                pareto_indices.append(idx)
                max_r = row['r']
        pareto_df = df_sorted.loc[pareto_indices].reset_index(drop=True)

        st.subheader("Best solutions")
        st.table(pareto_df.head()) 

        towrite_pareto = BytesIO()
        with pd.ExcelWriter(towrite_pareto, engine="xlsxwriter") as writer:
            pareto_df.to_excel(writer, index=False, sheet_name="Pareto Front")

        towrite_pareto.seek(0)

        st.download_button(
            label="Download solutions",
            data=towrite_pareto,
            file_name="pareto_solutions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Plotar o gráfico
        ax.scatter(df['a_c (m²)'], df['r'], color='blue', alpha=0.7)
        ax.plot(pareto_df['a_c (m²)'], pareto_df['r'], color='red', marker='o', linewidth=2)
        ax.set_title("Pareto front", fontsize=14)
        ax.set_xlabel("Cross section (m²)", fontsize=12)
        ax.set_ylabel("Prestressed level", fontsize=12)
        ax.grid(True)
        st.pyplot(fix)


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

# Seleção de modelo
model = st.radio(texts["model_label"], ["Ag"])# , ['Monte Carlo', "Ag"])

if model == 'Monte Carlo':
    st.subheader(texts["parameters"])
    g = st.number_input(texts["g_ext"], value=None)
    q = st.number_input(texts["q"], value=None)
    l = st.number_input(texts["l"], value=None)
    f_c = st.number_input(texts["f_c"], value=None)
    f_cj = st.number_input(texts["f_cj"], value=None)

    st.subheader(texts["algorithm_setup"])
    col1, col2 = st.columns(2)

    with col1:
        pres_min = st.number_input(texts["prestressed_min"], value=None)
        exc_min = st.number_input(texts["eccentricity_min"], value=None)
        width_min = st.number_input(texts["width_min"], value=None)
        height_min = st.number_input(texts["height_min"], value=None)
        pop_size = st.number_input(texts["pop_size"], value=None)

    with col2:
        pres_max = st.number_input(texts["prestressed_max"], value=None)
        exc_max = st.number_input(texts["eccentricity_max"], value=None)
        width_max = st.number_input(texts["width_max"], value=None)
        height_max = st.number_input(texts["height_max"], value=None)

    # Chamar função Monte Carlo (supondo que ela existe)
    # monte_carlo(g, q, l, f_c, f_cj, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)

elif model == "Ag":
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
        print(f"Executando simulação com os parâmetros selecionados...")
        ag_monte_carlo(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)