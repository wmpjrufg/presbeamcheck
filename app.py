from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla
from metapy_toolbox import metaheuristic_optimizer
from easyplot_toolbox import line_chart, histogram_chart, scatter_chart, bar_chart

import streamlit as st
import pandas as pd

st.title("Title")
st.header("Lore Ipsum Dolor Sit Amet Consectetur Adipiscing Elit")

st.subheader("Project Variables")

g = st.number_input('Dread load (kN/m)', value=None)
q = st.number_input('Live load (kN/m)', value=None)
l = st.number_input('Span (m)', value=None)
f_c = st.number_input('F_c (MPa)', value=None)
f_cj = st.number_input('F_cj (MPa)', value=None)

st.subheader("Algorithm Setup")

interations = st.number_input('Number of iterations', value=None)
pop_size = st.number_input('Population size', value=None)
pres_min = st.number_input('Prestressed mini', value=None)
pres_max = st.number_input('Prestressed max', value=None)
exc_min = st.number_input('Excentricity mini', value=None)
exc_max = st.number_input('Excentricity max', value=None)
width_min = st.number_input('Width mini', value=None)
width_max = st.number_input('Width max', value=None)
height_min = st.number_input('Height mini', value=None)
height_max = st.number_input('Height max', value=None)


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
        'number of iterations': int(interations),
        'number of population': int(pop_size),
        'number of dimensions': 4,
        'x pop lower limit': [pres_min, exc_min, width_min, height_min],
        'x pop upper limit': [pres_max, exc_max, width_min, height_max],
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
    st.write(df_all_reps[status])
    print(df_resume_all_reps[status])
    best_result_row = df_resume_all_reps[status].iloc[-1]
    of, g = new_obj_ic_jack_priscilla([best_result_row['X_0_BEST'], 
                                    best_result_row['X_1_BEST'], 
                                    best_result_row['X_2_BEST'], 
                                    best_result_row['X_3_BEST']], 
                                    variaveis_proj)
    result = ({
        'lambda': 0.5,
        'X_0_BEST': best_result_row['X_0_BEST'],
        'X_1_BEST': best_result_row['X_1_BEST'],
        'X_2_BEST': best_result_row['X_2_BEST'],
        'X_3_BEST': best_result_row['X_3_BEST'],
        'OF_0': of[0],
        'OF_1': of[1]
    })
    for i, g_value in enumerate(g):
        result[f'G_{i}'] = g_value

    results.append(result)

    df_results = pd.DataFrame(results)
    st.write(df_results)