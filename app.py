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
            st.write("Running Algorithm...")

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
                'flecha limite de fabrica (m)': 7/1000,
                'flecha limite de serviço (m)': 7/250
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


def monte_carlo(g, q, l, f_c, f_cj, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max):
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = int(pop_size)

    if st.button("Run Simulation"):
        st.write("Running Simulation...")
        np.random.seed(42)
        bw_samples = list(np.random.uniform(bw[0], bw[1], n))
        h_samples = list(np.random.uniform(h[0], h[1], n))
        p_samples = list(np.random.uniform(p[0], p[1], n))
        e_p_samples = list(np.random.uniform(e_p[0], e_p[1], n))
        bw_samples = list(np.random.uniform(bw[0], bw[1], n))
        h_samples = list(np.random.uniform(h[0], h[1], n))

        df = {'p (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples}
        df = pd.DataFrame(df)

        st.write(df)

        a_c_list = []
        r_list = []
        g_0_list = []
        g_1_list = []
        g_2_list = []

        fixed_variables = {
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
                        'flecha limite de fabrica (m)': 7/1000,
                        'flecha limite de serviço (m)': 7/250,
                    }
        
        for i, row in df.iterrows():
            of, g = new_obj_ic_jack_priscilla([row['p (kN)'], row['e_p (m)'], row['bw (m)'], row['h (m)']], fixed_variables)
            
            a_c_list.append(of[0])
            r_list.append(of[1])
            g_0_list.append(g[0])
            g_1_list.append(g[1])
            g_2_list.append(g[2])


        df['a_c (m²)'] = a_c_list
        df['r'] = r_list
        df['g_0'] = g_0_list
        df['g_1'] = g_1_list
        df['g_2'] = g_2_list
        df = pd.DataFrame(df)

        # Grafico com o rendimento
        # Filtrar linhas com g_0, g_1 e g_2 negativos
        df = df[(df['g_0'] <= 0) & (df['g_1'] <= 0) & (df['g_2'] <= 0)]

        # Mostrar as primeiras 10 linhas para verificar se as linhas foram removidas
        st.table(df.head(10))  # Exibe as primeiras 10 linhas

        # Salvando a planilha em um buffer (BytesIO)
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Simulação")

        towrite.seek(0)  

        st.download_button(
            label="Baixar Planilha Excel",
            data=towrite,
            file_name="simulacao_monte_carlo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Plotar gráfico de dispersão (scatter plot)
        fix, ax = plt.subplots()
        ax.scatter(df['a_c (m²)'], df['r'], color='blue', alpha=0.7)
        ax.set_title("Gráfico de Dispersão: Área da Seção (a_c) vs Coeficiente de Rendimento (r)", fontsize=14)
        ax.set_xlabel("Área da Seção (a_c) [m²]", fontsize=12)
        ax.set_ylabel("Coeficiente de Rendimento (r)", fontsize=12)
        ax.grid(True)
        st.pyplot(fix)


if __name__ == "__main__":
    st.title("Title")
    st.subheader("Project Variables")

    g = st.number_input('Dread load (kN/m)', value=None)
    q = st.number_input('Live load (kN/m)', value=None)
    l = st.number_input('Span (m)', value=None)
    f_c = st.number_input('F_c (MPa)', value=None)
    f_cj = st.number_input('F_cj (MPa)', value=None)

    st.subheader("Algorithm Setup")

    col1, col2 = st.columns(2)

    with col1:
        iterations = st.number_input('Number of iterations', value=None)
        pres_min = st.number_input('Prestressed mini', value=None)
        exc_min = st.number_input('Excentricity mini', value=None)
        width_min = st.number_input('Width mini', value=None)
        height_min = st.number_input('Height mini', value=None)
        

    with col2:
        pop_size = st.number_input('Population size', value=None)
        pres_max = st.number_input('Prestressed max', value=None)
        exc_max = st.number_input('Excentricity max', value=None)
        width_max = st.number_input('Width max', value=None)
        height_max = st.number_input('Height max', value=None)

    model = st.radio('Select Model', ['AG', 'Monte Carlo'])

    if model == 'AG':
         ag(g, q, l, f_c, f_cj, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)
    elif model == 'Monte Carlo':
        monte_carlo(g, q, l, f_c, f_cj, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)