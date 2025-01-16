"""Funções para problema de otimização multiobjetivo considerando uma viga protendida e pré-fabricadas"""
from typing import Any

import numpy as np

def propriedades_geometricas(geom: list, tipo: str) -> tuple:
    """
    Esta função determina as propriedades geométricas de seções transversais retangulares, I de abas paralelas e I de abas inclinadas.

    Argumentos:
        geom: Lista com as dimensões do perfil
              'retangular'        -> [bw=base, h=altura]
              'i abas paralelas'  -> []
              'i abas inclinadas' -> []
        tipo: Tipo do perfil ('retangular', 'i abas paralelas', 'i abas inclinadas')

    Retornos:
        a_c: Área da seção
        y_t: Distância centroíde para topo
        y_b: Distância centroíde para base
        i_c: Momento de inércia da seção
        w_t: Módulo resistente em relação ao topo
        w_b: Módulo resistente em relação à base
   """

    if tipo == 'retangular':
        base, altura = geom
        y_t = altura / 2
        y_b = altura / 2
        a_c = base * altura
        i_c = (base * altura**3) / 12
        w_t = i_c / y_t
        w_b = i_c / y_b

    # elif tipo == 'i tradicional':
    #     bf_t, hf_t, bw, h, bf_b, hf_b = geom
    #     a_c = (bf_t * hf_t) + (bw * h) + (bf_b * hf_b)       # Área total do perfil

    #     # Centroide
    #     y_c = ((((hf_t/2) + h + hf_b) * bf_t * hf_t) + (((h/2) + hf_b) * bw * h) + ((hf_b/2) * bf_b * hf_b)) / a_c    #Centroide em y

    #     # Momentos de Inércia em x
    #     i_x_aba_t = (((bf_t * hf_t ** 3) /12) + (bf_t * hf_t) * ((hf_t/2 + h + hf_b) - y_c)**2)
    #     i_x_aba_b = (((bf_b * hf_b ** 3) /12) + (bf_b * hf_b) * ((hf_b/2) - y_c)**2)
    #     i_x_alma = (((bw * h ** 3) /12) + (bw * h) * ((h/2 + hf_b) - y_c)**2)
    #     i_x = i_x_aba_t +  i_x_aba_b + i_x_alma             # Soma do momento de inércia

    #     # Módulos de resistência no topo e na base
    #     w_topo = w_base = i_x / y_c

    # elif tipo == 'i aba inclinadas':
    #     base_aba, altura, esp_alma, esp_aba, triangular = geom
    #     aba_inferior = base_aba - 2 * triangular  # Calcula a aba inferior com a medida da parte triangular
    #     a_aba = (base_aba + aba_inferior) / 2 * esp_aba  # Área da aba inclinada
    #     a_alma = (altura - 2 * esp_aba) * esp_alma       # Área da alma
    #     a_c = 2 * a_aba + a_alma                         # Área total

    #     # Centroide
    #     y_c = altura / 2  # Por simetria

    #     # Momentos de inércia
    #     i_x_aba_ind = (esp_aba**3 / 36) * (base_aba**2 + 4 * base_aba * aba_inferior + aba_inferior**2) / (base_aba + aba_inferior)
    #     i_x_aba = 2 * (i_x_aba_ind)
    #     i_x_alma = (esp_alma * (altura - 2 * esp_aba)**3) / 12
    #     i_x = i_x_aba + i_x_alma

    #     # Módulos de resistência no topo e na base
    #     w_topo = w_base = i_x / (altura / 2)

    return a_c, y_t, y_b, i_c, w_t, w_b


def rigidez(l: float, e_c: float, i_c: float, p: float=1.0) -> float:
    """
    Esta função determina a rigidez da seção transversal da peça de concreto considerando uma carga pontual no meio do vão.

    Argumentos:
        l: Comprimento do vão (m)
        e_c: Módulo de elasticidade (kN/m2)
        i_c: Momento de inércia da seção bruta (m4)
        p: Força no meio da viga (kN)

    Retornos:
        delta: rigidez da seção transversal
    """
    
    return (48 * e_c * i_c) / (p * l ** 3)


def grau_protensao_i(e_p: float, m_g: float, m_q: float, p: float, w_b: float, a_c: float) -> float:
    m_p = p * e_p
    m_o = p / a_c * w_b + m_p
    m_t = m_g + m_q
    return m_o / m_t


def grau_protensao_ii(e_p: float, a_c: float, g_ext: float, p: float, l: float) -> float:
    g = a_c * 25
    return p * e_p * 8 / ((g + g_ext) * l ** 2)


def flecha_biapoiada_carga_distribuida(l: float, e_c: float, i_c: float, w: float) -> float:
    """
    Está função determina a flecha de uma viga biapoiada com carga distribuída
    
    Argumentos:
        l: Comprimento do vão (m)
        e_c: Módulo de elasticidade (kN/m2)
        i_c: Momento de inércia da seção bruta (m4)
        w: Carga distribuída (kN/m)

    Retornos:
        delta: flecha da viga
    """
    return (5 * w * l ** 4) / (384 * e_c * i_c)


def flecha_biapoiada_carga_protensao(l: float, e_c: float, i_c: float, m_p: float) -> float:
    """
    Está função determina a flecha de uma viga biapoiada com carga distribuída
    
    Argumentos:
        l: Comprimento do vão (m)
        e_c: Módulo de elasticidade (kN/m2)
        i_c: Momento de inércia da seção bruta (m4)
        m_p: Momento fletor na seção (kNm)

    Retornos:
        delta: flecha da viga
    """
    return (m_p * l ** 2) / (8 * e_c * i_c)


def modulo_elasticidade_concreto(agregado: float, f_ck: float, f_ckj: float, impressao: bool=False) -> tuple:
    """
    Esta função calcula os módulos de elasticidade do concreto em diferentes idades.
    
    Argumentos:
        agregado: Tipo de agregado usado no traço do cimento
                    'BAS' - agregado de Basalto
                    'GRA' - agregado de Granito
                    'CAL' - agregado de Calcário
                    'ARE' - agregado de Arenito
        f_ck: Resistência característica à compressão (kN/m2)
        f_ckj: Resistência característica à compressão idade j (kN/m2)
        impressao: Critério de impressão dos resultados para visualização

    Retornos:
        e_cij: Módulo de elasticidade tangente em uma idade j dias (kN/m2)
        e_csj: Módulo de elasticidade secante em uma idade j dias (kN/m2)
    """
    
    # Determinação do módulo tangente e_ci idade T
    if agregado.upper() == 'BAS':
        alfa_e = 1.2
    elif agregado.upper() == 'GRA':
        alfa_e = 1.0
    elif agregado.upper() == 'CAL':
        alfa_e = 0.9
    elif agregado.upper() == 'ARE':
        alfa_e = 0.7
    
    # Módulo tangente e_ci idade 28 dias
    f_ck /= 1E3
    if f_ck <= 50:
        e_ci = alfa_e * 5600 * np.sqrt(f_ck)
    elif f_ck > 50:
        e_ci = 21.5 * (10 ** 3) * alfa_e * (f_ck / 10 + 1.25) ** (1 / 3)
    alfa_i = 0.8 + 0.2 * f_ck / 80
    if alfa_i > 1:
        alfa_i = 1
    
    # Determinação do módulo secante e_cs idade 28 dias
    e_ci *= 1E3
    e_cs = e_ci * alfa_i
    
    # Determinação dos módulos de elasticidade em um idade J
    if f_ck <= 50:
        f_ck *= 1E3
        e_cij = e_ci * (f_ckj / f_ck) ** 0.5
    elif f_ck > 50:
        f_ck *= 1E3
        e_cij = e_ci * (f_ckj / f_ck) ** 0.3
    e_csj = e_cij * alfa_i
    
    # Impressões
    if impressao == True:
        print("\n")
        print("PROPRIEDADES DE RIGIDEZ DO MATERIAL")
        print("e_ci 28 dias: %.3E kN/m²" % e_ci)
        print("e_cs 28 dias: %.3E kN/m²" % e_cs)
        print("e_cij idade j dias: %.3E kN/m²" % e_cij)
        print("e_csj idade j dias: %.3E kN/m²" % e_csj)
        print("\n")

    return e_ci, e_cs, e_cij, e_csj


def esforcos_bi_apoiada(w: float, l: float) -> tuple:
    """
    Esta função determina os esforços em uma viga bi-apoiada
    
    Argumentos:
        w: Carga distribuída (kN/m)
        l: Comprimento do vão (m)

    Retornos:
        m_max: Momento máximo (kN.m)
        v_max: Força cortante máxima (kN)
    """
    
    m_max = (w * l ** 2) / 8
    v_max = w * l / 2
    
    return m_max, v_max


def resistencia_concreto(f_ck: float, f_ckj: float, impressao: bool=False):
    """
    Esta função determina propriedades do concreto em uma idade j.
    
    Entrada:
    f_ck       | Resistência característica à compressão                 | kN/m²  | float   
    TEMPO      | Tempo                                                   | dias   | float
    CIMENTO    | Cimento utilizado                                       |        | string    
               |   'CP1' - Cimento portland 1                            |        | 
               |   'CP2' - Cimento portland 2                            |        |              
               |   'CP3' - Cimento portland 3                            |        |
               |   'CP4' - Cimento portland 4                            |        | 
               |   'CP5' - Cimento portland 5                            |        | 
    AGREGADO   | Tipo de agragado usado no traço do cimento              |        | string    
               |   'BAS' - Agregado de Basalto                           |        | 
               |   'GRA' - Agregado de Granito                           |        |              
               |   'CAL' - Agregado de Calcário                          |        |
               |   'ARE' - Agregado de Arenito                           |        | 
    PRINT      | Critério de impressão dos resultados para visualização  |        | string
    
    Saída:
    f_ckJ      | Resistência característica à compressão idade J         | kN/m²  | float
    f_ctmj     | Resistência média caracteristica a tração idade J       | kN/m²  | float
    F_CTKINFJ  | Resistência média caracteristica a tração inf idade J   | kN/m²  | float
    F_CTKSUPJ  | Resistência média caracteristica a tração sup idade J   | kN/m²  | float
    E_CIJ      | Módulo de elasticidade tangente                         | kN/m²  | float
    E_CSJ      | Módulo de elasticidade do secante                       | kN/m²  | float      
    """
    
    # Propriedades em situação de tração f_ct em uma idade de 28 dias
    f_ck /= 1E3
    if f_ck <= 50:
        # Condição classe inferior ou igual a C50
        f_ctm = 0.3 * f_ck ** (2/3)
    elif f_ck > 50:
        # Condição classe superior a C50
        f_ctm = 2.12 * np.log(1 + 0.10 * (f_ck + 8))
    f_ctm *= 1E3
    f_ctkinf = 0.7 * f_ctm 
    f_ctksup = 1.3 * f_ctm
    
    # Propriedades em situação de tração f_ct em uma idade de j dias
    f_ckj /= 1E3
    if f_ck <= 50:
        # Condição classe inferior ou igual a C50
        f_ctmj = 0.3 * f_ckj ** (2/3)
    elif f_ck > 50:
        # Condição classe superior a C50
        f_ctmj = 2.12 * np.log(1 + 0.10 * (f_ck + 8))
    f_ctmj *= 1E3
    f_ctkinfj = 0.7 * f_ctmj 
    f_ctksupj = 1.3 * f_ctmj

    return  f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup


def obj_ic_jack_priscilla(x: float, none_variable: Any):
    """
    Esta função determina o valor da função pseudoobjetivo.

    Argumentos:
        x: Lista com as variáveis de decisão
        none_variable: Qualquer valor ou quaisquers valores que desejam-se otimizar

    Retornos:
        of: função objetivo
    """
    
    # Variáveis de entrada
    p = x[0]
    e_p = x[1]
    b_w = x[2]
    h = x[3]
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    tipo_protensao = none_variable['tipo de protensão']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    lambdaa = none_variable['lambda']
    rp = none_variable['penalidade']
    phi = none_variable['fator de fluência']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']

    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)

    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)
    rig = rigidez(l, e_cs, i_c)

    # Esforços
    m_gext, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdato = 1.00 * (m_gpp)
    m_sdserv = 1.00 * (m_gext + 0.6 * m_q)

    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)

    of = lambdaa * (a_c / 4) - (1 - lambdaa) * kappa_p

    # Tensão no topo e na base na transferência da protensão considerando que as perdas de protensão são de 5%
    p_sd_ato = 1.10 * (0.95 * p)
    sigma_t_ato_mv = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_sdato / w_t)
    sigma_t_ato_ap = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b_ato_mv = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_sdato / w_b)
    sigma_b_ato_ap = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    if tipo_protensao == "Parcial":
        sigma_max_trac = 1.20 * f_ctmj
        f_ck /= 1E3
        if f_ck <= 50:
            sigma_max_comp = 0.70 * f_ck_ato
        else:
            f_ck_ato /= 1E3
            sigma_max_comp = (0.70 * (1 - (f_ck_ato - 50) / 200)) * 1E3
            f_ck_ato *= 1E3
        f_ck *= 1E3

    # Tensão no topo e na base considerando que as perdas de protensão são de 20%
    p_sd_serv = 1.10 * (0.80 * p)
    sigma_t1 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_sdato / w_t) + (m_sdserv / w_t)
    sigma_t2 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b1 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_sdato / w_b) - (m_sdserv / w_b)
    sigma_b2 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Cálculos da equação de estado limite para tensões elásticas no topo da seção
    g = []
    if sigma_t_ato_mv < 0:
        sigma_t_ato_mv = abs(sigma_t_ato_mv)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t1 / sigma_max - 1)

    if sigma_t_ato_ap < 0:
        sigma_t_ato_ap = abs(sigma_t_ato_ap)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t_ato_ap / sigma_max - 1)

    if sigma_b_ato_mv < 0:
        sigma_b_ato_mv = abs(sigma_b_ato_mv)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b_ato_mv / sigma_max - 1)

    if sigma_b_ato_ap < 0:
        sigma_b_ato_ap = abs(sigma_b_ato_ap)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b_ato_ap / sigma_max - 1)
    
    # Limites de tensão com base no tipo de protensão
    if tipo_protensao == "Parcial":
        sigma_max_trac = f_ctmj
        sigma_max_comp = 0.70 * f_ck_ato

    # Cálculos da equação de estado limite para tensões elásticas no topo da seção
    if sigma_t1 < 0:
        sigma_t1 = abs(sigma_t1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t1 / sigma_max - 1)

    if sigma_t2 < 0:
        sigma_t2 = abs(sigma_t2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t2 / sigma_max - 1)

    if sigma_b1 < 0:
        sigma_b1 = abs(sigma_b1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b1 / sigma_max - 1)

    if sigma_b2 < 0:
        sigma_b2 = abs(sigma_b2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b2 / sigma_max - 1)

    # Restrição de flecha no ato
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs, i_c, p)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext)
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, 0.6 * q)
    g.append((phi * delta_ato + phi * delta_serv_0 + delta_serv_1) / delta_lim_serv - 1)

    # Restrição de flecha inicial
    g.append(delta_lim_serv * 0.25 / (phi * delta_ato + phi * delta_serv_0 + delta_serv_1) - 1)

    # Restrição construtiva
    g.append(e_p / (h * 0.90)  - 1)

    # Restrição de esbeltez
    g.append(2 / (l / h) - 1)

    # Restrição de instabilidade
    g.append((l / 50) / b_w - 1)
    g.append((0.2 * h) / b_w - 1)

    for i in g:
        of += rp * max(0, i) ** 2

    return of


def new_obj_ic_jack_priscilla(x: float, none_variable: Any) -> tuple[list, list]:
    """
    Esta função determina o valor da função pseudoobjetivo.

    Argumentos:
        x: Lista com as variáveis de decisão
        none_variable: Qualquer valor ou quaisquers valores que desejam-se otimizar

    Retornos:
        of: função objetivo
    """
    
    # Variáveis de entrada
    p = x[0]
    e_p = x[1]
    b_w = x[2]
    h = x[3]
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    tipo_protensao = none_variable['tipo de protensão']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    lambdaa = none_variable['lambda']
    rp = none_variable['penalidade']
    phi = none_variable['fator de fluência']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']
    
    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)
    
    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)    
    rig = rigidez(l, e_cs, i_c)
 
    # Esforços
    m_g, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdato = 1.00 * (m_gpp)
    m_sdserv = 1.00 * (m_g + m_gpp + 0.6 * m_q)

    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)

    of = [a_c, kappa_p]

    # Tensão no topo e na base na transferência da protensão considerando que as perdas de protensão são de 5%
    p_sd_ato = 1.10 * (0.95 * p)
    sigma_t1 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_sdato / w_t)
    sigma_t2 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b1 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_sdato / w_b)
    sigma_b2 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    if tipo_protensao == "Parcial":
        sigma_max_trac = f_ctmj
        sigma_max_comp = 0.70 * f_ck_ato

    # Cálculos da equação de estado limite para tensões elásticas no topo da seção
    g = []
    if sigma_t1 < 0:
        sigma_t1 = abs(sigma_t1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t1 / sigma_max - 1)

    if sigma_t2 < 0:
        sigma_t2 = abs(sigma_t2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t2 / sigma_max - 1)

    if sigma_b1 < 0:
        sigma_b1 = abs(sigma_b1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b1 / sigma_max - 1)

    if sigma_b2 < 0:
        sigma_b2 = abs(sigma_b2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b2 / sigma_max - 1)

    # Tensão no topo e na base considerando que as perdas de protensão são de 20%
    p_sd_ato = 1.10 * (0.80 * p)
    sigma_t1 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_sdato / w_t) + (m_sdserv / w_t)
    sigma_t2 = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b1 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_sdato / w_b) - (m_sdserv / w_b)
    sigma_b2 = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    if tipo_protensao == "Parcial":
        sigma_max_trac = f_ctmj
        sigma_max_comp = 0.70 * f_ck_ato

    # Cálculos da equação de estado limite para tensões elásticas no topo da seção
    if sigma_t1 < 0:
        sigma_t1 = abs(sigma_t1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t1 / sigma_max - 1)

    if sigma_t2 < 0:
        sigma_t2 = abs(sigma_t2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_t2 / sigma_max - 1)

    if sigma_b1 < 0:
        sigma_b1 = abs(sigma_b1)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b1 / sigma_max - 1)

    if sigma_b2 < 0:
        sigma_b2 = abs(sigma_b2)
        sigma_max = sigma_max_trac
    else:
        sigma_max = sigma_max_comp
    g.append(sigma_b2 / sigma_max - 1)

    # Restrição de flecha no ato
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs, i_c, p)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext)
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, 0.6 * q)
    g.append((phi * delta_ato + phi * delta_serv_0 + delta_serv_1) / delta_lim_serv - 1)

    # Restrição de flecha inicial
    g.append(delta_lim_serv * 0.25 / (phi * delta_ato + phi * delta_serv_0 + delta_serv_1) - 1)

    # Restrição construtiva
    g.append(e_p / (h * 0.90)  - 1)

    # Restrição de esbeltez
    g.append(2 / (l / h) - 1)

    # Restrição de instabilidade
    g.append((l / 50) / b_w - 1)
    g.append((0.2 * h) / b_w - 1)

    return of, g


# def multi_obj_0(x, none_variable):
#     x_0 = x[0]
#     of_0 = x_0 ** 2
#     of_1 = (x_0 - 2) ** 2
#     lambdaa = none_variable['lambda']
#     of = lambdaa * of_0 + (1 - lambdaa) * of_1
#     return of


# def new_multi_obj_0(x, none_variable):
#     x_0 = x[0]
#     of_0 = x_0 ** 2
#     of_1 = (x_0 - 2) ** 2
#     of = [of_0, of_1]
#     return of


# def multi_obj_1(x, none_variable):
#     x_0 = x[0]
#     x_1 = x[1]
#     x_2 = x[2]

#     of_0 = -10 * np.exp(-0.2 * np.sqrt(x_0**2 + x_1**2)) \
#            -10 * np.exp(-0.2 * np.sqrt(x_1**2 + x_2**2))
    
#     of_1 = (abs(x_0)**0.8 + 5 * np.sin(x_0**3)) + \
#            (abs(x_1)**0.8 + 5 * np.sin(x_1**3)) + \
#            (abs(x_2)**0.8 + 5 * np.sin(x_2**3))

#     lambdaa = none_variable['lambda']
#     of = lambdaa * of_0 + (1 - lambdaa) * of_1
#     return of


# def new_multi_obj_1(x, none_variable):
#     x_0 = x[0]
#     x_1 = x[1]
#     x_2 = x[2]

#     of_0 = -10 * np.exp(-0.2 * np.sqrt(x_0**2 + x_1**2)) \
#            -10 * np.exp(-0.2 * np.sqrt(x_1**2 + x_2**2))
    
#     of_1 = (abs(x_0)**0.8 + 5 * np.sin(x_0**3)) + \
#            (abs(x_1)**0.8 + 5 * np.sin(x_1**3)) + \
#            (abs(x_2)**0.8 + 5 * np.sin(x_2**3))

#     of = [of_0, of_1]
#     return of


# def multi_obj_2(x, none_variable):
#     const = 1 / np.sqrt(len(x))
#     of_0 = 1 - np.exp(-sum((x_i - const)**2 for x_i in x))
#     of_1 = 1 - np.exp(-sum((x_i + const)**2 for x_i in x))

#     lambdaa = none_variable['lambda']
#     of = lambdaa * of_0 + (1 - lambdaa) * of_1
#     return of


# def new_multi_obj_2(x, none_variable):
#     const = 1 / np.sqrt(len(x))
#     of_0 = 1 - np.exp(-sum((x_i - const)**2 for x_i in x))
#     of_1 = 1 - np.exp(-sum((x_i + const)**2 for x_i in x))

#     of = [of_0, of_1]
#     return of

p = 360.0
e_p = 0.319
b_w = 0.20
h = 0.85
variaveis_proj = {
                        'g (kN/m)': 10.75,
                        'q (kN/m)': 5.00,
                        'l (m)': 12,
                        'tipo de seção': 'retangular',
                        'tipo de protensão': 'Parcial',
                        'fck,ato (kPa)': 24.56E3,
                        'fck (kPa)': 30E3,
                        'lambda': 0.50,
                        'penalidade': 1E6,
                        'fator de fluência': 2.5,
                        'flecha limite de fabrica (m)': 7/1000,
                        'flecha limite de serviço (m)': 7/250
                    }
obj_ic_jack_priscilla([p, e_p, b_w, h], variaveis_proj)

# if __name__ == '__main__':

#     p = 423.20
#     e_p = 0.319
#     b_w = 0.20
#     h = 0.80
#     propriedades_geometricas([b_w, h], 'retangular')
#     # b_w = 1.77
#     # h = 1.77
#     # l = 7.00
#     # lam=0.01
#     # tipo = 'retangular'
#     # e_ci, e_cs, _, _ = modulo_elasticidade_concreto('gra', 50000, 21000, True)
#     # print(resistencia_concreto(50000, 21000, True))
#     # a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)    
#     # rig = rigidez(l, e_cs, i_c)
#     # print(a_c, rig)
#     # print(lam*a_c/4+(1-lam)*rig/6834403.501427528)
#     # variaveis_proj = {
#     #                     'g (kN/m)': 7.5,
#     #                     'q (kN/m)': 7.5,
#     #                     'l (m)': 7,
#     #                     'tipo de seção': 'retangular',
#     #                     'tipo de protensão': 'Parcial',
#     #                     'fck,ato (kPa)': 21E3,
#     #                     'fck (kPa)': 50E3,
#     #                     'lambda': 0.5,
#     #                     'penalidade': 1E6,
#     #                     'fator de fluência inicial': 0.5,
#     #                     'fator de fluência final': 2.5,
#     #                     'flecha limite de fabrica (m)': 0.003,
#     #                     'flecha limite de serviço (m)': 0.005
#     #                 }
#     # print(obj_ic_jack_priscilla([1, 1, 1, 1], variaveis_proj))
#     # g_ext = 10.75
#     # e_p = 0.319
#     # p = 423.20
#     # l = 12.0
#     # print(grau_protensao_ii(e_p, 0.20*0.85, g_ext, p, l))
