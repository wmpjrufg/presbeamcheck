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
            g_lists.append(g)


        df['a_c (m²)'] = a_c_list
        df['r'] = r_list

        for idx, g_list in enumerate(zip(*g_lists)):
            df[f'g_{idx}'] = g_list

        df = pd.DataFrame(df)

        # Grafico com o rendimento
        df = df[(df[[col for col in df.columns if col.startswith('g_')]] <= 0).all(axis=1)]
        df.reset_index(drop=True, inplace=True)
        st.subheader("Simulation results")
        st.table(df.head(10)) 


        # Salvando a planilha em um buffer (BytesIO)
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Simulação")

        towrite.seek(0)  

        st.download_button(
            label="Download results",
            data=towrite,
            file_name="simulacao_monte_carlo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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
        st.table(pareto_df.head(10)) 

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
        ax.set_xlabel("Cross section [m²]", fontsize=12)
        ax.set_ylabel("Obj 1 (r)", fontsize=12)
        ax.grid(True)
        st.pyplot(fix)


if __name__ == "__main__":
    st.title("Title")
    st.subheader("Project Variables")
    st.write("""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam id suscipit mauris. Etiam ultricies tellus at lobortis posuere. Nulla a eros id lacus finibus imperdiet nec nec risus. 
                Nam eget placerat justo, vitae molestie lorem. Donec lacus nisl, fringilla ac risus eu, egestas dapibus turpis. Pellentesque faucibus volutpat nibh sed tempus. Nulla pulvinar mattis rhoncus.""") 
                
    
    model = st.radio('Select Model', ['AG', 'Monte Carlo'])

    g = st.number_input('Dead load (kN/m)', value=None)
    q = st.number_input('Live load (kN/m)', value=None)
    l = st.number_input('Beam load (m)', value=None)
    f_c = st.number_input('fc (MPa)', value=None)
    f_cj = st.number_input('fcj (MPa)', value=None)

    st.subheader("Algorithm Setup")

    if model == 'AG':
        col1, col2 = st.columns(2)

        with col1:
            iterations = st.number_input('Number of iterations', value=None)
            pres_min = st.number_input('Prestressed minimum', value=None)
            exc_min = st.number_input('Excentricity minimum', value=None)
            width_min = st.number_input('Width minimum', value=None)
            height_min = st.number_input('Height minimum', value=None)
                

        with col2:
            pop_size = st.number_input('Population size', value=None)
            pres_max = st.number_input('Prestressed maximum', value=None)
            exc_max = st.number_input('Excentricity maximum', value=None)
            width_max = st.number_input('Width maximum', value=None)
            height_max = st.number_input('Height maximum', value=None)
            
        ag(g, q, l, f_c, f_cj, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)

    elif model == 'Monte Carlo':
        col1, col2 = st.columns(2)

        with col1:
            pres_min = st.number_input('Prestressed minimum', value=None)
            exc_min = st.number_input('Excentricity minimum', value=None)
            width_min = st.number_input('Width minimum', value=None)
            height_min = st.number_input('Height minimum', value=None)
            pop_size = st.number_input('Number of samples', value=None)
                
        with col2:
            pres_max = st.number_input('Prestressed maximum', value=None)
            exc_max = st.number_input('Excentricity maximum', value=None)
            width_max = st.number_input('Width maximum', value=None)
            height_max = st.number_input('Height maximum', value=None)

        monte_carlo(g, q, l, f_c, f_cj, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)