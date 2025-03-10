"""Funções para problema de otimização multiobjetivo considerando uma viga protendida e pré-fabricadas"""
from typing import Any

import numpy as np

def propriedades_geometricas(geom: list, tipo: str, impressao: bool=False) -> tuple:
    """Esta função determina as propriedades geométricas de seções transversais retangulares, I de abas paralelas e I de abas inclinadas.

    Argumentos:
        geom: Lista com as dimensões do perfil
              'retangular'        -> [bw=base, h=altura]
              'i abas paralelas'  -> []
              'i abas inclinadas' -> []
        tipo: Tipo do perfil ('retangular', 'i abas paralelas', 'i abas inclinadas')

    Retornos:
        a_c: Área da seção
        y_t: Distância centroide para topo
        y_b: Distância centroide para base
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

    if impressao == True:
        print("\n")
        print("PROPRIEDADES GEOMÉTRICAS")
        print("Área da seção: %.2f m²" % a_c)
        print("Distância centroide para topo: %.2f m" % y_t)
        print("Distância centroide para base: %.2f m" % y_b)
        print("Momento de inércia da seção: %.2f m^4" % i_c)
        print("Módulo resistente em relação ao topo: %.2f m^3" % w_t)
        print("Módulo resistente em relação à base: %.2f m^3" % w_b)
        print("\n")

    return a_c, y_t, y_b, i_c, w_t, w_b


def grau_protensao_ii(e_p: float, a_c: float, g_ext: float, p: float, l: float, impressao: bool=False) -> float:
    """Está função determina o grau de protensão de uma viga protendida.

    Args:
        e_p (float): Excentricidade da protensão (m)
        a_c (float): Área da seção transversal (m²)
        g_ext (float): Carga externa (kN/m)
        p (float): Carga de protensão (kN)
        l (float): Comprimento do vão (m)

    Returns:
        Beta (float): Grau de protensão
    """
    g = a_c * 25

    if impressao == True:
        print("\n")
        print("GRAU DE PROTENSÃO")
        print("Carga de protensão: %.2f kN" % p)
        print("Carga externa: %.2f kN/m" % g_ext)
        print("Comprimento do vão: %.2f m" % l)
        print("Área da seção transversal: %.2f m²" % a_c)
        print("Excentricidade da protensão: %.2f m" % e_p)
        print("Grau de protensão: %.2f" % (p * e_p * 8 / (g * l ** 2)))
        print("\n")

    return p * e_p * 8 / ((g + g_ext) * l ** 2)


def flecha_biapoiada_carga_distribuida(l: float, e_c: float, i_c: float, w: float, impressao: bool=False) -> float:
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

    if impressao == True:
        print("\n")
        print("FLECHA BI-APOIADA")
        print("Comprimento do vão: %.2f m" % l)
        print("Módulo de elasticidade: %.2f kN/m²" % e_c)
        print("Momento de inércia da seção bruta: %.2f m^4" % i_c)
        print("Carga distribuída: %.2f kN/m" % w)
        print("Flecha da viga: %.2f m" % ((5 * w * l ** 4) / (384 * e_c * i_c)))
        print("\n")

    return (5 * w * l ** 4) / (384 * e_c * i_c)


def flecha_biapoiada_carga_protensao(l: float, e_c: float, i_c: float, m_p: float, impressao: bool=False) -> float:
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

    if impressao == True:
        print("\n")
        print("FLECHA BI-APOIADA")
        print("Comprimento do vão: %.2f m" % l)
        print("Módulo de elasticidade: %.2f kN/m²" % e_c)
        print("Momento de inércia da seção bruta: %.2f m^4" % i_c)
        print("Momento fletor na seção: %.2f kNm" % m_p)
        print("Flecha da viga: %.2f m" % ((m_p * l ** 2) / (8 * e_c * i_c)))
        print("\n")

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


def esforcos_bi_apoiada(w: float, l: float, impressao: bool=False) -> tuple:
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
    
    if impressao == True:
        print("\n")
        print("ESFORÇOS EM UMA VIGA BI-APOIADA")
        print("Carga distribuída: %.2f kN/m" % w)
        print("Comprimento do vão: %.2f m" % l)
        print("Momento máximo: %.2f kN.m" % m_max)
        print("Força cortante máxima: %.2f kN" % v_max)
        print("\n")
        
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

    # Impressões
    if impressao == True:
        print("\n")
        print("PROPRIEDADES DE RIGIDEZ DO MATERIAL")
        print("f_ctm 28 dias: %.3E kN/m²" % f_ctm)
        print("f_ctkinf 28 dias: %.3E kN/m²" % f_ctkinf)
        print("f_ctksup 28 dias: %.3E kN/m²" % f_ctksup)
        print("f_ctmj idade j dias: %.3E kN/m²" % f_ctmj)
        print("f_ctkinfj idade j dias: %.3E kN/m²" % f_ctkinfj)
        print("f_ctksupj idade j dias: %.3E kN/m²" % f_ctksupj)
        print("\n")

    return  f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup


def prop_geometrica_estadio_ii(h_f, b_f, b_w, a_st, a_sc, alpha_mod, d, d_l, impressao=False):
    """
    Esta função calcula as propriedades geométricas de uma peça de concreto armado no estádio II.

    Entrada:
    H_F        | Altura de mesa superior da seção Tê                     | m    | float
    b_f        | Base de mesa superior da seção Tê                       | m    | float
    b_w        | Base de alma da seção Tê                                | m    | float
    a_st       | Área de aço na seção tracionada                         | m²   | float
    a_sc       | Área de aço na seção comprimida                         | m²   | float
    ALPHA_MOD  | Relação entre os módulos de elasticidade aço-concreto   |      | float
    D          | Altura útil da armadura tracionada                      | m    | float
    d_l        | Altura útil da armadura comprimida                      | m    | float
    PRINT      | Critério de impressão dos resultados para visualização  |      | string

    Saída:
    X_II       | Centro geometrico da viga no estádio 2                  | m    | float
    i_ii       | Inércia da viga no estádio 2                            | m^4  | float
    """
    
    # Checagem do primeiro teste de linha neutra. Hipótese de b_w = b_f
    a_1 = b_f / 2
    h_faux = h_f * 0
    a_2 = h_faux * (b_f - b_w) + (alpha_mod - 1) * a_sc + alpha_mod * a_st
    a_3 = - d_l * (alpha_mod - 1) * a_sc - d * alpha_mod * a_st - (h_faux ** 2 / 2) * (b_f - b_w)
    x_iiteste = (- a_2 + (a_2 ** 2 - 4 * a_1 * a_3) ** 0.50) / (2 * a_1)
    
    # Cálculo da linha neutra em função do teste de 1º chute
    if x_iiteste <= h_f or h_f == 0:
        # L.N. passando pela mesa
        x_ii = x_iiteste
        pasa_onde = "mesa"
    elif x_iiteste > h_f:
        # L.N. passando pela alma
        a_1 = b_w / 2
        a_2 = h_f * (b_f - b_w) + (alpha_mod - 1) * a_sc + alpha_mod * a_st
        a_3 = - d_l * (alpha_mod - 1) * a_sc - d * alpha_mod * a_st - (h_f ** 2 / 2) * (b_f - b_w)
        x_ii = (- a_2 + (a_2 ** 2 - 4 * a_1 * a_3) ** 0.50) / (2 * a_1)
        pasa_onde = "alma"
    
    # Inércia estádio II
    if x_ii <= h_f or h_f == 0:
        # L.N. passando pela mesa
        i_ii = (b_f * x_ii ** 3) / 3 + alpha_mod * a_st * (x_ii - d) ** 2 + (alpha_mod - 1) * a_sc * (x_ii - d_l) ** 2
    else:
        # L.N. passando pela alma
        i_ii = ((b_f - b_w) * h_f ** 3) / 12 + (b_w * x_ii ** 3) / 3 + (b_f - b_w) * (x_ii - h_f * 0.50) ** 2 + alpha_mod * a_st * (x_ii - D) ** 2 + (alpha_mod - 1) * a_sc * (x_ii - d_l) ** 2
   
    # Impressões
    if impressao == True:
        print("\n")
        print("PROP. ESTÁDIO II")
        print("a_1: ", a_1)
        print("a_2: ", a_2)
        print("a_3: ", a_3)
        print("x_ii: ", x_ii)
        print("i_ii: ", i_ii)
        print("Passa pela: ", pasa_onde)
        print("x_iiteste: ", x_iiteste)
        print("\n")

    return x_ii, i_ii, a_1, a_2, a_3, pasa_onde, x_iiteste


def fissuracao(e_cs: float, h:float, b_w: float, i_c: float, y_b: float, m_sd: float, f_ctd: float, p_sd: float, w_klim: float, impressao: bool=False) -> float:
    """
    """
    a_c = b_w * h
    e_s = 210E6
    alpha_e = e_s / e_cs
    d = 0.88 * h
    x_ii, i_ii, _, _, _, _, _ = prop_geometrica_estadio_ii(h_f=0, b_f=b_w, b_w=b_w, a_st=0.20/100*b_w*h, a_sc=0, alpha_mod=alpha_e, d=d, d_l=0, impressao=False)
    # m_r = 1.5 * f_cd * (i_c / y_b) 
    # m_rbranson = (m_r / m_sd) ** 3
    # i_eq = m_rbranson * i_c + (1 - m_rbranson) * i_ii
    envolv = 7.5 * 12.5 / 1000
    d_linha = h - d
    a_cri = b_w * (d_linha + envolv)
    rho_r = (0.20/100*b_w*h) / a_cri
    sigma_s = -alpha_e * (p_sd / a_c) + alpha_e * ((m_sd * (d - x_ii)) / i_ii)
    phi = 12.5 / 1000
    n_1 = 2.25
    w_11 = phi / (12.5 * n_1)
    w_12 = sigma_s / e_s
    w_13 = 3 * sigma_s / f_ctd
    w_1 = w_11 * w_12 * w_13
    #w_2 = (phi / (12.5 * n_1)) * (sigma_s / e_s) * (4 / rho_r + 45)
    g = w_1 / w_klim - 1

    if impressao == True:
        print("\n")
        print("Fissuração")
        print("a_c: ", a_c)
        print("alpha_e: ", alpha_e)
        print("d: ", d)
        print("x_ii: ", x_ii)
        print("i_ii: ", i_ii)
        print("envolv: ", envolv)
        print("d_linha: ", d_linha)
        print("a_cri: ", a_cri)
        print("sigma_s: ", sigma_s)
        print("w_1: ", w_1)
        print("g: ", g)
        print("\n")

    return w_1, g


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
    g = []
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    eta = none_variable['lambda']
    rp = none_variable['rp']
    phi_a = none_variable['fator de fluência para o ato']
    phi_b = none_variable['fator de fluência para o serviço']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']
    phi_els = none_variable['coeficiente parcial para carga q']
    perda_inicial = none_variable['perda inicial de protensão (%)'] / 100
    perda_total = none_variable['perda total de protensão (%)'] / 100
    # w_klim = none_variable['abertura fissura limite (m)']

    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)

    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)

    # Esforços
    m_gext, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdserv = m_gext + phi_els * m_q

    # Função objetivo que maximiza a compensação de protensão
    # of = lambdaa * (a_c / 0.18) - (1 - lambdaa) * kappa_p
    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)
    of = -kappa_p

    # Tensão no topo e na base na transferência da protensão considerando as perdas iniciais de protensão
    p_sd_ato = 1.10 * ((1 - perda_inicial) * p)
    m_psd_ato = p_sd_ato * e_p
    sigma_t_ato_mv = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_gpp / w_t)
    sigma_t_ato_ap = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b_ato_mv = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_gpp / w_b)
    sigma_b_ato_ap = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_trac = -1.20 * f_ctmj
    f_ck /= 1E3
    f_ck_ato /= 1E3
    if f_ck <= 50:
        sigma_max_comp = 0.70 * f_ck_ato
        sigma_max_comp *= 1E3
    else:
        sigma_max_comp = (0.70 * (1 - (f_ck_ato - 50) / 200))
        sigma_max_comp *= 1E3
    f_ck *= 1E3
    f_ck_ato *= 1E3

    # Restrição de seção transversal para segunda função objetivo
    g.append(np.abs(a_c - eta)/0.001 - 1)                                 # g_0

    # Restrição de limite de tensão ft <= sigma <= fc
    g.append(sigma_t_ato_mv / sigma_max_comp - 1)                         # g_1
    g.append((sigma_max_trac - sigma_t_ato_mv) / abs(sigma_max_trac))     # g_2
    g.append(sigma_t_ato_ap / sigma_max_comp - 1)                         # g_3   
    g.append((sigma_max_trac - sigma_t_ato_ap) / abs(sigma_max_trac))     # g_4
    g.append(sigma_b_ato_mv / sigma_max_comp - 1)                         # g_5
    g.append((sigma_max_trac - sigma_b_ato_mv) / abs(sigma_max_trac))     # g_6
    g.append(sigma_b_ato_ap / sigma_max_comp - 1)                         # g_7
    g.append((sigma_max_trac - sigma_b_ato_ap) / abs(sigma_max_trac))     # g_8

    # Tensão no topo e na base no serviço considerando as perdas totais de protensão
    p_sd_serv = 1.10 * ((1 - perda_total) * p)
    sigma_t_serv_mv = (p_sd_serv / a_c) - (p_sd_serv * e_p / w_t) + (m_gpp / w_t) + (m_sdserv / w_t)
    sigma_b_serv_mv = (p_sd_serv / a_c) + (p_sd_serv * e_p / w_b) - (m_gpp / w_b) - (m_sdserv / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_comp = 0.60 * f_ck
    sigma_max_trac = -1.50 * f_ctkinf

    # Restrição de limite de tensão ft <= sigma <= 0, para base
    g.append(sigma_b_serv_mv/abs(sigma_max_trac))                          # g_9
    g.append((sigma_max_trac - sigma_b_serv_mv) / abs(sigma_max_trac))     # g_10
    
    # Restrição de limite de tensão 0 <= sigma <= fc, para base
    g.append(sigma_t_serv_mv / sigma_max_comp - 1)                         # g_11   
    g.append(-sigma_t_serv_mv / sigma_max_comp)                            # g_12

    # Restrição de flecha no armazenamento
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs_ato, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs_ato, i_c, m_psd_ato)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)                    # g_13

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext) 
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, phi_els * q)
    delta_total = phi_a * delta_ato  + phi_b * (delta_serv_0 + delta_serv_1)
    g.append(delta_total / delta_lim_serv - 1)                             # g_14

    # Restrição construtiva
    g.append(e_p / (0.90 * 0.50 * h)  - 1)                                 # g_15

    # Restrição de esbeltez 18.3.1
    g.append(2 / (l / h) - 1)                                              # g_16

    # Restrição de largura máxima
    g.append(b_w / (h * 0.50) - 1)                                         # g_17

    # Restrição de instabilidade 15.10
    g.append((l / 50) / b_w - 1)                                           # g_18

    # # Restriçã de fissuração
    # if sigma_b_ato_mv_serv < 0:
    #     if abs(sigma_b_ato_mv_serv) < 1.5 * f_ctm:
    #         g.append(sigma_b_ato_mv_serv / (-1.5 * f_ctm) - 1)
    #     else:
    #         g.append(# se maior que 1.5 fctm aplica cirterio)
    #         # # fissuração
    #         # m_sdf = m_sdato + m_sdserv
    #         # _, g_fiss = fissuracao(e_cs, h, b_w, i_c, y_b, m_sdf, f_ctm, p_sd_serv, w_klim)
    #         # g.append(g_fiss) #g_13
    # else:
    #     g.append(sigma_b_ato_mv_serv / sigma_max_comp - 1)

    # Função pseudo-objetivo 
    for i in g:
        of += rp * max(0, i) ** 2

    return of


def obj_ic_jack_priscilla_sobol(x: float, none_variable: Any):
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
    g = []
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    eta = none_variable['lambda']
    rp = none_variable['rp']
    phi_a = none_variable['fator de fluência para o ato']
    phi_b = none_variable['fator de fluência para o serviço']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']
    phi_els = none_variable['coeficiente parcial para carga q']
    perda_inicial = none_variable['perda inicial de protensão (%)'] / 100
    perda_total = none_variable['perda total de protensão (%)'] / 100
    # w_klim = none_variable['abertura fissura limite (m)']

    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)

    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)

    # Esforços
    m_gext, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdserv = m_gext + phi_els * m_q

    # Função objetivo que maximiza a compensação de protensão
    # of = lambdaa * (a_c / 0.18) - (1 - lambdaa) * kappa_p
    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)
    of = -kappa_p

    # Tensão no topo e na base na transferência da protensão considerando as perdas iniciais de protensão
    p_sd_ato = 1.10 * ((1 - perda_inicial) * p)
    m_psd_ato = p_sd_ato * e_p
    sigma_t_ato_mv = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_gpp / w_t)
    sigma_t_ato_ap = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b_ato_mv = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_gpp / w_b)
    sigma_b_ato_ap = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_trac = -1.20 * f_ctmj
    f_ck /= 1E3
    f_ck_ato /= 1E3
    if f_ck <= 50:
        sigma_max_comp = 0.70 * f_ck_ato
        sigma_max_comp *= 1E3
    else:
        sigma_max_comp = (0.70 * (1 - (f_ck_ato - 50) / 200))
        sigma_max_comp *= 1E3
    f_ck *= 1E3
    f_ck_ato *= 1E3

    # Restrição de seção transversal para segunda função objetivo
    g.append(np.abs(a_c - eta)/0.001 - 1)                                 # g_0

    # Restrição de limite de tensão ft <= sigma <= fc
    g.append(sigma_t_ato_mv / sigma_max_comp - 1)                         # g_1
    g.append((sigma_max_trac - sigma_t_ato_mv) / abs(sigma_max_trac))     # g_2
    g.append(sigma_t_ato_ap / sigma_max_comp - 1)                         # g_3   
    g.append((sigma_max_trac - sigma_t_ato_ap) / abs(sigma_max_trac))     # g_4
    g.append(sigma_b_ato_mv / sigma_max_comp - 1)                         # g_5
    g.append((sigma_max_trac - sigma_b_ato_mv) / abs(sigma_max_trac))     # g_6
    g.append(sigma_b_ato_ap / sigma_max_comp - 1)                         # g_7
    g.append((sigma_max_trac - sigma_b_ato_ap) / abs(sigma_max_trac))     # g_8

    # Tensão no topo e na base no serviço considerando as perdas totais de protensão
    p_sd_serv = 1.10 * ((1 - perda_total) * p)
    sigma_t_serv_mv = (p_sd_serv / a_c) - (p_sd_serv * e_p / w_t) + (m_gpp / w_t) + (m_sdserv / w_t)
    sigma_b_serv_mv = (p_sd_serv / a_c) + (p_sd_serv * e_p / w_b) - (m_gpp / w_b) - (m_sdserv / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_comp = 0.60 * f_ck
    sigma_max_trac = -1.50 * f_ctkinf

    # Restrição de limite de tensão ft <= sigma <= 0, para base
    g.append(sigma_b_serv_mv/abs(sigma_max_trac))                          # g_9
    g.append((sigma_max_trac - sigma_b_serv_mv) / abs(sigma_max_trac))     # g_10
    
    # Restrição de limite de tensão 0 <= sigma <= fc, para base
    g.append(sigma_t_serv_mv / sigma_max_comp - 1)                         # g_11   
    g.append(-sigma_t_serv_mv / sigma_max_comp)                            # g_12

    # Restrição de flecha no armazenamento
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs_ato, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs_ato, i_c, m_psd_ato)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)                    # g_13

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext) 
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, phi_els * q)
    delta_total = phi_a * delta_ato  + phi_b * (delta_serv_0 + delta_serv_1)
    g.append(delta_total / delta_lim_serv - 1)                             # g_14

    # Restrição construtiva
    g.append(e_p / (0.90 * 0.50 * h)  - 1)                                 # g_15

    # Restrição de esbeltez 18.3.1
    g.append(2 / (l / h) - 1)                                              # g_16

    # Restrição de largura máxima
    g.append(b_w / (h * 0.50) - 1)                                         # g_17

    # Restrição de instabilidade 15.10
    g.append((l / 50) / b_w - 1)                                           # g_18

    # # Restriçã de fissuração
    # if sigma_b_ato_mv_serv < 0:
    #     if abs(sigma_b_ato_mv_serv) < 1.5 * f_ctm:
    #         g.append(sigma_b_ato_mv_serv / (-1.5 * f_ctm) - 1)
    #     else:
    #         g.append(# se maior que 1.5 fctm aplica cirterio)
    #         # # fissuração
    #         # m_sdf = m_sdato + m_sdserv
    #         # _, g_fiss = fissuracao(e_cs, h, b_w, i_c, y_b, m_sdf, f_ctm, p_sd_serv, w_klim)
    #         # g.append(g_fiss) #g_13
    # else:
    #     g.append(sigma_b_ato_mv_serv / sigma_max_comp - 1)

    # Função pseudo-objetivo 
    for i in g:
        of += rp * max(0, i) ** 2

    return [None], [None], [of]


def new_obj_ic_jack_priscilla(x: float, none_variable: Any):
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
    g = []
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    phi_a = none_variable['fator de fluência para o ato']
    phi_b = none_variable['fator de fluência para o serviço']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']
    phi_els = none_variable['coeficiente parcial para carga q']
    perda_inicial = none_variable['perda inicial de protensão (%)'] / 100
    perda_total = none_variable['perda total de protensão (%)'] / 100
    # w_klim = none_variable['abertura fissura limite (m)']

    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)

    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)

    # Esforços
    m_gext, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdserv = m_gext + phi_els * m_q

    # Função objetivo que maximiza a compensação de protensão
    # of = lambdaa * (a_c / 0.18) - (1 - lambdaa) * kappa_p
    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)
    of = [a_c, kappa_p]

    # Tensão no topo e na base na transferência da protensão considerando as perdas iniciais de protensão
    p_sd_ato = 1.10 * ((1 - perda_inicial) * p)
    m_psd_ato = p_sd_ato * e_p
    sigma_t_ato_mv = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_gpp / w_t)
    sigma_t_ato_ap = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b_ato_mv = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_gpp / w_b)
    sigma_b_ato_ap = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_trac = -1.20 * f_ctmj
    f_ck /= 1E3
    f_ck_ato /= 1E3
    if f_ck <= 50:
        sigma_max_comp = 0.70 * f_ck_ato
        sigma_max_comp *= 1E3
    else:
        sigma_max_comp = (0.70 * (1 - (f_ck_ato - 50) / 200))
        sigma_max_comp *= 1E3
    f_ck *= 1E3
    f_ck_ato *= 1E3

    # Restrição de seção transversal para segunda função objetivo
    #g.append(np.abs(a_c - eta)/0.001 - 1)                                 # g_0

    # Restrição de limite de tensão ft <= sigma <= fc
    g.append(sigma_t_ato_mv / sigma_max_comp - 1)                         # g_1
    g.append((sigma_max_trac - sigma_t_ato_mv) / abs(sigma_max_trac))     # g_2
    g.append(sigma_t_ato_ap / sigma_max_comp - 1)                         # g_3   
    g.append((sigma_max_trac - sigma_t_ato_ap) / abs(sigma_max_trac))     # g_4
    g.append(sigma_b_ato_mv / sigma_max_comp - 1)                         # g_5
    g.append((sigma_max_trac - sigma_b_ato_mv) / abs(sigma_max_trac))     # g_6
    g.append(sigma_b_ato_ap / sigma_max_comp - 1)                         # g_7
    g.append((sigma_max_trac - sigma_b_ato_ap) / abs(sigma_max_trac))     # g_8

    # Tensão no topo e na base no serviço considerando as perdas totais de protensão
    p_sd_serv = 1.10 * ((1 - perda_total) * p)
    sigma_t_serv_mv = (p_sd_serv / a_c) - (p_sd_serv * e_p / w_t) + (m_gpp / w_t) + (m_sdserv / w_t)
    sigma_b_serv_mv = (p_sd_serv / a_c) + (p_sd_serv * e_p / w_b) - (m_gpp / w_b) - (m_sdserv / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_comp = 0.60 * f_ck
    sigma_max_trac = -1.50 * f_ctkinf

    # Restrição de limite de tensão ft <= sigma <= 0, para base
    g.append(sigma_b_serv_mv/abs(sigma_max_trac))                          # g_9
    g.append((sigma_max_trac - sigma_b_serv_mv) / abs(sigma_max_trac))     # g_10
    
    # Restrição de limite de tensão 0 <= sigma <= fc, para base
    g.append(sigma_t_serv_mv / sigma_max_comp - 1)                         # g_11   
    g.append(-sigma_t_serv_mv / sigma_max_comp)                            # g_12

    # Restrição de flecha no armazenamento
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs_ato, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs_ato, i_c, m_psd_ato)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)                    # g_13

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext) 
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, phi_els * q)
    delta_total = phi_a * delta_ato  + phi_b * (delta_serv_0 + delta_serv_1)
    g.append(delta_total / delta_lim_serv - 1)                             # g_14

    # Restrição construtiva
    g.append(e_p / (0.90 * 0.50 * h)  - 1)                                 # g_15

    # Restrição de esbeltez 18.3.1
    g.append(2 / (l / h) - 1)                                              # g_16

    # Restrição de largura máxima
    g.append(b_w / (h * 0.50) - 1)                                         # g_17

    # Restrição de instabilidade 15.10
    g.append((l / 50) / b_w - 1)                                           # g_18

    # # Restriçã de fissuração
    # if sigma_b_ato_mv_serv < 0:
    #     if abs(sigma_b_ato_mv_serv) < 1.5 * f_ctm:
    #         g.append(sigma_b_ato_mv_serv / (-1.5 * f_ctm) - 1)
    #     else:
    #         g.append(# se maior que 1.5 fctm aplica cirterio)
    #         # # fissuração
    #         # m_sdf = m_sdato + m_sdserv
    #         # _, g_fiss = fissuracao(e_cs, h, b_w, i_c, y_b, m_sdf, f_ctm, p_sd_serv, w_klim)
    #         # g.append(g_fiss) #g_13
    # else:
    #     g.append(sigma_b_ato_mv_serv / sigma_max_comp - 1)

    return of, g


def new_obj_ic_jack_pris_html(x: float, none_variable: Any):
    # Variáveis de entrada
    p = x[0]
    e_p = x[1]
    b_w = x[2]
    h = x[3]
    g = []
    tipo = none_variable['tipo de seção']
    g_ext = none_variable['g (kN/m)']
    q = none_variable['q (kN/m)']
    l = none_variable['l (m)']
    f_ck_ato = none_variable['fck,ato (kPa)']
    f_ck = none_variable['fck (kPa)']
    phi_a = none_variable['fator de fluência para o ato']
    phi_b = none_variable['fator de fluência para o serviço']
    delta_lim_fabrica = none_variable['flecha limite de fabrica (m)']
    delta_lim_serv = none_variable['flecha limite de serviço (m)']
    phi_els = none_variable['coeficiente parcial para carga q']
    perda_inicial = none_variable['perda inicial de protensão (%)'] / 100
    perda_total = none_variable['perda total de protensão (%)'] / 100

    # Propriedades do material
    e_ci, e_cs, e_ci_ato, e_cs_ato = modulo_elasticidade_concreto('gra', f_ck, f_ck_ato, False)
    f_ctmj, f_ctkinfj, f_ctksupj, f_ctm, f_ctkinf, f_ctksup = resistencia_concreto(f_ck, f_ck_ato, False)

    # Propriedades geométricas
    # Problema retangular
    a_c, y_t, y_b, i_c, w_t, w_b = propriedades_geometricas([b_w, h], tipo)

    # Esforços
    m_gext, _ = esforcos_bi_apoiada(g_ext, l)
    m_q, _ = esforcos_bi_apoiada(q, l)
    g_pp = a_c * 25
    m_gpp, _ = esforcos_bi_apoiada(g_pp, l)
    m_sdserv = m_gext + phi_els * m_q

    # Função objetivo que maximiza a compensação de protensão
    kappa_p = grau_protensao_ii(e_p, a_c, g_ext, p, l)
    of = [a_c, kappa_p]

    # Tensão no topo e na base na transferência da protensão considerando as perdas iniciais de protensão
    p_sd_ato = 1.10 * ((1 - perda_inicial) * p)
    m_psd_ato = p_sd_ato * e_p
    sigma_t_ato_mv = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t) + (m_gpp / w_t)
    sigma_t_ato_ap = (p_sd_ato / a_c) - (p_sd_ato * e_p / w_t)
    sigma_b_ato_mv = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b) - (m_gpp / w_b)
    sigma_b_ato_ap = (p_sd_ato / a_c) + (p_sd_ato * e_p / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_trac = -1.20 * f_ctmj
    f_ck /= 1E3
    f_ck_ato /= 1E3
    if f_ck <= 50:
        sigma_max_comp = 0.70 * f_ck_ato
        sigma_max_comp *= 1E3
    else:
        sigma_max_comp = (0.70 * (1 - (f_ck_ato - 50) / 200))
        sigma_max_comp *= 1E3
    f_ck *= 1E3
    f_ck_ato *= 1E3

    # Restrição de limite de tensão ft <= sigma <= fc
    g.append(sigma_t_ato_mv / sigma_max_comp - 1)                         # g_1
    g.append((sigma_max_trac - sigma_t_ato_mv) / abs(sigma_max_trac))     # g_2
    g.append(sigma_t_ato_ap / sigma_max_comp - 1)                         # g_3   
    g.append((sigma_max_trac - sigma_t_ato_ap) / abs(sigma_max_trac))     # g_4
    g.append(sigma_b_ato_mv / sigma_max_comp - 1)                         # g_5
    g.append((sigma_max_trac - sigma_b_ato_mv) / abs(sigma_max_trac))     # g_6
    g.append(sigma_b_ato_ap / sigma_max_comp - 1)                         # g_7
    g.append((sigma_max_trac - sigma_b_ato_ap) / abs(sigma_max_trac))     # g_8

    # Tensão no topo e na base no serviço considerando as perdas totais de protensão
    p_sd_serv = 1.10 * ((1 - perda_total) * p)
    sigma_t_serv_mv = (p_sd_serv / a_c) - (p_sd_serv * e_p / w_t) + (m_gpp / w_t) + (m_sdserv / w_t)
    sigma_b_serv_mv = (p_sd_serv / a_c) + (p_sd_serv * e_p / w_b) - (m_gpp / w_b) - (m_sdserv / w_b)

    # Limites de tensão com base no tipo de protensão
    sigma_max_comp = 0.60 * f_ck
    sigma_max_trac = -1.50 * f_ctkinf

    # Restrição de limite de tensão ft <= sigma <= 0, para base
    g.append(sigma_b_serv_mv/abs(sigma_max_trac))                          # g_9
    g.append((sigma_max_trac - sigma_b_serv_mv) / abs(sigma_max_trac))     # g_10
    
    # Restrição de limite de tensão 0 <= sigma <= fc, para base
    g.append(sigma_t_serv_mv / sigma_max_comp - 1)                         # g_11   
    g.append(-sigma_t_serv_mv / sigma_max_comp)                            # g_12

    # Restrição de flecha no armazenamento
    delta_ato_0 = flecha_biapoiada_carga_distribuida(l, e_cs_ato, i_c, g_pp)
    delta_ato_1 = flecha_biapoiada_carga_protensao(l, e_cs_ato, i_c, m_psd_ato)
    delta_ato = delta_ato_0 + (-delta_ato_1)
    g.append(np.abs(delta_ato) / delta_lim_fabrica - 1)                    # g_13

    # Restrição de flecha no serviço
    delta_serv_0 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, g_ext) 
    delta_serv_1 = flecha_biapoiada_carga_distribuida(l, e_cs, i_c, phi_els * q)
    delta_total = phi_a * delta_ato  + phi_b * (delta_serv_0 + delta_serv_1)
    g.append(delta_total / delta_lim_serv - 1)                             # g_14

    # Restrição construtiva
    g.append(e_p / (0.90 * 0.50 * h)  - 1)                                 # g_15

    # Restrição de esbeltez 18.3.1
    g.append(2 / (l / h) - 1)                                              # g_16

    # Restrição de largura máxima
    g.append(b_w / (h * 0.50) - 1)                                         # g_17

    # Restrição de instabilidade 15.10
    g.append((l / 50) / b_w - 1)                                           # g_18

    # Criar conteúdo HTML
    html_content = f"""
    <html>
    <head><title>Resultados</title></head>
    <body>
    <h1>Propriedades dos Materiais</h1>
    <ul>
        <li>e_ci: {e_ci}</li>
        <li>e_cs: {e_cs}</li>
        <li>e_ci_ato: {e_ci_ato}</li>
        <li>e_cs_ato: {e_cs_ato}</li>
        <li>f_ctmj: {f_ctmj}</li>
        <li>f_ctkinfj: {f_ctkinfj}</li>
        <li>f_ctksupj: {f_ctksupj}</li>
        <li>f_ctm: {f_ctm}</li>
        <li>f_ctkinf: {f_ctkinf}</li>
        <li>f_ctksup: {f_ctksup}</li>
    </ul>
    
    <h1>Propriedades Geométricas</h1>
    <ul>
        <li>a_c: {a_c}</li>
        <li>y_t: {y_t}</li>
        <li>y_b: {y_b}</li>
        <li>i_c: {i_c}</li>
        <li>w_t: {w_t}</li>
        <li>w_b: {w_b}</li>
    </ul>
    
    <h1>Esforços</h1>
    <ul>
        <li>m_gext: {m_gext}</li>
        <li>m_q: {m_q}</li>
        <li>g_pp: {g_pp}</li>
        <li>m_gpp: {m_gpp}</li>
        <li>m_sdserv: {m_sdserv}</li>
    </ul>
    
    <h1>Funções Objetivo</h1>
    <ul>
        <li>f1 = {a_c}</li>
        <li>f2 = {kappa_p}</li>
    </ul>
    
    <h1>Tensões ato da Protensão</h1>
    <ul>
        <li>sigma_t_ato_mv: {sigma_t_ato_mv}</li>
        <li>sigma_t_ato_ap: {sigma_t_ato_ap}</li>
        <li>sigma_b_ato_mv: {sigma_b_ato_mv}</li>
        <li>sigma_b_ato_ap: {sigma_b_ato_ap}</li>
        <li>sigma_max_comp: {sigma_max_comp}</li>
        <li>sigma_max_trac: {sigma_max_trac}</li>
    </ul>
    
    <h1>Restrições</h1>
    <ul>
        {''.join([f'<li>g{i+1}: {g[i]}</li>' for i in range(18)])}
    </ul>
    
    </body>
    </html>
    """
    
    return html_content