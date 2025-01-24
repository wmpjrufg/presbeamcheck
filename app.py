from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla
from metapy_toolbox import metaheuristic_optimizer
from easyplot_toolbox import line_chart, histogram_chart, scatter_chart, bar_chart
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ag(g, q, l, f_c, f_cj, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max):
    if st.button("Run Algorithm"):
            variaveis_proj = {
                'g (kN/m)': g,
                'q (kN/m)': q,
                'l (m)': l,
                'tipo de seção': 'retangular',
                'tipo de protensão': 'Parcial',
                'fck,ato (kPa)': f_c * 1E3,
                'fck (kPa)': f_cj * 1E3,
                'lambda': 0.5,
                'penalidade': 1E6,
                'fator de fluência': 2.5,
                'flecha limite de fabrica (m)': l/1000,
                'flecha limite de serviço (m)': l/250
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
                                        'crossover': {'crossover rate (%)': 82, 'type':'linear'},
                                        'mutation': {'mutation rate (%)': 12, 'type': 'hill climbing', 'cov (%)': 15, 'pdf': 'gaussian'},
                                        }
            }
            
            results = []

            general_setup = {   
                        'number of repetitions': 30,
                        'type code': 'real code',
                        'initial pop. seed': [None] * 30,
                        'algorithm': 'genetic_algorithm_01',
                    }
            
            df_all_reps, df_resume_all_reps, reports, status = metaheuristic_optimizer(algorithm_setup, general_setup)
            st.table(df_all_reps[status])


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


def change_language(lang):
    if lang == "en":
        return {
            "title": "Prestressed Beam Check Routine",
            "description": """This app checks a simple supported beam subject to one dead and live load. 
                              The user needs to fill in a project variables interval (prestressed load, eccentricity, width, and height). 
                              The algorithm checks linear stress when a prestressed load is introduced in the beam, and it also checks 
                              linear stress in service and the geometric constraints of ABNT NBR 6118.""",
            "model_label": "Select Model",
            "dead_load": "Dead load (kN/m)",
            "live_load": "Live load (kN/m)",
            "beam_length": "Beam length (m)",
            "fc": "fc (MPa)",
            "fcj": "fcj (MPa)",
            "algorithm_setup": "Algorithm Setup",
            "iterations": "Number of iterations",
            "prestressed_min": "Prestressed minimum",
            "eccentricity_min": "Eccentricity minimum",
            "width_min": "Width minimum",
            "height_min": "Height minimum",
            "population_size": "Population size",
            "prestressed_max": "Prestressed maximum",
            "eccentricity_max": "Eccentricity maximum",
            "width_max": "Width maximum",
            "height_max": "Height maximum",
            "samples": "Number of samples"
        }
    else:  # pt
        return {
            "title": "Rotina de Verificação de Viga Protendida",
            "description": """Este aplicativo verifica uma viga biapoiada sujeita a uma carga permanente e acidental. 
                              O usuário precisa preencher o intervalo das variáveis do projeto (carga protendida, excentricidade, largura e altura). 
                              O algoritmo verifica a tensão linear quando a carga protendida é introduzida na viga, e também verifica 
                              a tensão linear em serviço e as restrições geométricas da ABNT NBR 6118.""",
            "model_label": "Selecione o Modelo",
            "dead_load": "Carga permanente (kN/m)",
            "live_load": "Carga acidental (kN/m)",
            "beam_length": "Comprimento da viga (m)",
            "fc": "fck (MPa)",
            "fcj": "fcj (MPa)",
            "algorithm_setup": "Configuração do Algoritmo",
            "iterations": "Número de iterações",
            "prestressed_min": "Carga protendida mínima",
            "eccentricity_min": "Excentricidade mínima",
            "width_min": "Largura mínima",
            "height_min": "Altura mínima",
            "population_size": "Tamanho da população",
            "prestressed_max": "Carga protendida máxima",
            "eccentricity_max": "Excentricidade máxima",
            "width_max": "Largura máxima",
            "height_max": "Altura máxima",
            "samples": "Número de amostras"
        }


if __name__ == "__main__":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("&#127463;&#127479; Português"):  # 🇧🇷
            lang = "pt"
        else:
            lang = "en"
    with col2:
        if st.button("&#127468;&#127463; English"):  # 🇬🇧
            lang = "en"

    texts = change_language(lang)
    st.title(texts["title"])
    st.write(texts["description"])

    # Seleção de modelo
    model = st.radio(texts["model_label"], ['Monte Carlo'])

    # Entradas principais
    g = st.number_input(texts["dead_load"], value=None)
    q = st.number_input(texts["live_load"], value=None)
    l = st.number_input(texts["beam_length"], value=None)
    f_c = st.number_input(texts["fc"], value=None)
    f_cj = st.number_input(texts["fcj"], value=None)

    st.subheader(texts["algorithm_setup"])

    if model == 'Monte Carlo':
        col1, col2 = st.columns(2)

        with col1:
            pres_min = st.number_input(texts["prestressed_min"], value=None)
            exc_min = st.number_input(texts["eccentricity_min"], value=None)
            width_min = st.number_input(texts["width_min"], value=None)
            height_min = st.number_input(texts["height_min"], value=None)
            pop_size = st.number_input(texts["samples"], value=None)

        with col2:
            pres_max = st.number_input(texts["prestressed_max"], value=None)
            exc_max = st.number_input(texts["eccentricity_max"], value=None)
            width_max = st.number_input(texts["width_max"], value=None)
            height_max = st.number_input(texts["height_max"], value=None)

        # Função de Monte Carlo
        monte_carlo(g, q, l, f_c, f_cj, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)
        