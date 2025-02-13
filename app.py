from protendido import obj_ic_jack_priscilla, new_obj_ic_jack_priscilla
from metapy_toolbox import metaheuristic_optimizer
from easyplot_toolbox import line_chart, histogram_chart, scatter_chart, bar_chart
from io import BytesIO
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def ag_monte_carlo(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max):
    import pandas as pd
    n_lambda = 20      
    n_length = 20000    
    p = [pres_min, pres_max]
    e_p = [exc_min, exc_max]
    bw = [width_min, width_max]
    h = [height_min, height_max]
    n = n_length

    np.random.seed(42)
    p_samples = list(np.random.uniform(p[0], p[1], n))
    e_p_samples = list(np.random.uniform(e_p[0], e_p[1], n))
    bw_samples = list(np.random.uniform(bw[0], bw[1], n))
    h_samples = list(np.random.uniform(h[0], h[1], n))

    df = {'p (kN)': p_samples, 'e_p (m)': e_p_samples, 'bw (m)': bw_samples, 'h (m)': h_samples}
    df = pd.DataFrame(df)
    
    a_c_list = []
    r_list = []
    rig_list = []
    g_lists = []

    for i, row in df.iterrows():
        fixed_variables = {
                            'g (kN/m)': g_ext,
                            'q (kN/m)': q,
                            'l (m)': l,
                            'tipo de seção': 'retangular',
                            'fck,ato (kPa)': f_cj * 1E3,
                            'fck (kPa)': f_c * 1E3,
                            'fator de fluência para o ato': phi_a,
                            'fator de fluência para o serviço': phi_b,
                            'flecha limite de fabrica (m)': l/1000,
                            'flecha limite de serviço (m)': l/250,
                            'coeficiente parcial para carga q': psi,
                            'perda inicial de protensão (%)': perda_inicial,
                            'perda total de protensão (%)': perda_final
                            }

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
    
    ac_min = float(df['a_c (m²)'].min())
    ac_max = float(df['a_c (m²)'].max())

    import pandas as pd
    lambda_list = list(np.linspace(ac_min, ac_max, n_lambda))
    results = []
    iter_var = 0

    for lambda_value in lambda_list:
        print(f'Iteration: {iter_var}, Lambda: {lambda_value}')

        variaveis_proj = {
                                'g (kN/m)': g_ext,
                                'q (kN/m)': q,
                                'l (m)': l,
                                'tipo de seção': 'retangular',
                                'fck,ato (kPa)': f_cj * 1E3,
                                'fck (kPa)': f_c * 1E3,
                                'lambda': lambda_value,
                                'rp': 1E6,
                                'fator de fluência para o ato': phi_a,
                                'fator de fluência para o serviço': phi_b,
                                'flecha limite de fabrica (m)': l/1000,
                                'flecha limite de serviço (m)': l/250,
                                'coeficiente parcial para carga q': psi,
                                'perda inicial de protensão (%)': perda_inicial,
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
                                                        'crossover': {'crossover rate (%)': 90, 'type':'linear'},
                                                        'mutation': {'mutation rate (%)': 20, 'type': 'hill climbing', 'cov (%)': 10, 'pdf': 'gaussian'},
                                                        }
                        }

        general_setup = {   
                            'number of repetitions': 15,
                            'type code': 'real code',
                            'initial pop. seed': [None] * 15,
                            'algorithm': 'genetic_algorithm_01',
                        }
        
        df_all_reps, df_resume_all_reps, reports, status = metaheuristic_optimizer(algorithm_setup, general_setup)
        best_result_row = df_resume_all_reps[status].iloc[-1]
        of, g = new_obj_ic_jack_priscilla([best_result_row['X_0_BEST'], 
                                        best_result_row['X_1_BEST'], 
                                        best_result_row['X_2_BEST'], 
                                        best_result_row['X_3_BEST']], 
                                        variaveis_proj)
        result = {
                    'lambda': lambda_value,
                    'X_0_BEST': best_result_row['X_0_BEST'],
                    'X_1_BEST': best_result_row['X_1_BEST'],
                    'X_2_BEST': best_result_row['X_2_BEST'],
                    'X_3_BEST': best_result_row['X_3_BEST'],
                    'OF_0': of[0],
                    'OF_1': of[1]
                }
        for i, g_value in enumerate(g):
            result[f'G_{i}'] = g_value
        iter_var += 1
        results.append(result)
    df_results = pd.DataFrame(results)

    # Gerando a figura 
    fig, ax = plt.subplots()
    ax.scatter(df_results['OF_0'], df_results['OF_1'], color='red', label='Fronteira eficiente')
    ax.scatter(df['a_c (m²)'], df['r'], color='#dcdcdc', label='Monte Carlo')

    ax.set_xlabel('Área da seção (m²)', fontsize=14)
    ax.set_ylabel('Carga $g$ estabilizada (%)', fontsize=14)
    ax.grid(False)
    ax.legend()

    # Exibindo os resultados
    st.subheader("Results")
    st.write(df_results)
    st.pyplot(fig)

    towrite_pareto = BytesIO()
    with pd.ExcelWriter(towrite_pareto, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Pareto Front")

    towrite_pareto.seek(0)

    st.download_button(
        label="Download solutions",
        data=towrite_pareto,
        file_name="ag_solutions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


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
    if lang == "pt":
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
            "g_ext": "Carga externa permanente (kN/m)",
            "q": "Carga variável de utilização (kN/m)",
            "l": "Vão da viga (m)",
            "f_c": "Resistência característica à compressão no serviço (MPa)",
            "f_cj": "Resistência característica à compressão no ato (MPa)",
            "phi_a": "Coeficiente de fluência para carregamento no ato",
            "phi_b": "Coeficiente de fluência para carregamento no serviço",
            "psi": "Coeficiente ψ redutor para ação variável",
            "perda_inicial": "Estimativa pecentual da perda inicial de protensão (%)",
            "perda_final": "Estimativa pecentual da perda total de protensão (%)",
            "parameters": "Parâmetros da viga protendida",
            "algorithm_setup": "Configuração do Algoritmo",
            "iterations": "Número de iterações",
            "prestressed_min": "Carga de protensão (kN) - valor inferior",
            "prestressed_max": "Carga de protensão (kN) - valor superior",
            "eccentricity_min": "Excentricidade de protensão (m) - valor inferior",
            "eccentricity_max": "Excentricidade de protensão (m) - valor superior",
            "width_min": "Largura da seção (m) - valor inferior",
            "width_max": "Largura da seção (m) - valor superior",
            "height_min": "Altura da seção (m) - valor inferior",
            "height_max": "Altura da seção (m) - valor superior",
            "pop_size": "Número de agentes para busca",
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
    model = st.radio(texts["model_label"], ['Monte Carlo', "Ag"])

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

        # Função de Monte Carlo
        monte_carlo(g, q, l, f_c, f_cj, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)
    
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

        if st.button("Run Simulation"):
            print(f"g_ext: {g_ext}, q: {q}, l: {l}, f_c: {f_c}, f_cj: {f_cj}, phi_a: {phi_a}, phi_b: {phi_b}, psi: {psi}, perda_inicial: {perda_inicial}, perda_final: {perda_final}, iterations: {iterations}, pop_size: {pop_size}, pres_min: {pres_min}, pres_max: {pres_max}, exc_min: {exc_min}, exc_max: {exc_max}, width_min: {width_min}, width_max: {width_max}, height_min: {height_min}, height_max: {height_max}")
            ag_monte_carlo(g_ext, q, l, f_c, f_cj, phi_a, phi_b, psi, perda_inicial, perda_final, iterations, pop_size, pres_min, pres_max, exc_min, exc_max, width_min, width_max, height_min, height_max)


