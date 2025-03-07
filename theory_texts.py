import streamlit as st

def texto_01():
    st.title("Projeto de peças pré-fabricadas e protendidas")

    # Introdução
    st.write("""
    Esta seção apresenta os principais procedimentos adotados para a verificação e o pré-dimensionamento de peças dessa natureza. 
    Como o foco deste trabalho está no pré-dimensionamento e nas verificações iniciais, foram consideradas as conclusões de 
    [Moraes et al.](https://doi.org/10.1590/S1983-41952022000600006) como referência. Nesse contexto, destaca-se que as restrições 
    de flecha e a verificação das tensões, tanto no ato da protensão quanto em serviço, são fatores determinantes para o controle 
    do dimensionamento estrutural de elementos pré-fabricados e protendidos. A seguir, são apresentadas algumas diretrizes normativas 
    para essas verificações.

    Para o projeto das peças pré-fabricadas e protendidas, foram adotadas as normas ABNT NBR 
    [[ABNT NBR 6118:2023](ISBN 978-85-5634-035-9)] e [[ABNT NBR 9062:2017](ISBN 978-85-5634-024-3)].
    """)

    st.header("Verificação de tensões")

    st.write("""
    As verificações de tensões em peças de concreto protendido seguem princípios semelhantes aos do concreto armado. No entanto, 
    a análise deve considerar tanto a fase de aplicação inicial da protensão quanto as condições de serviço subsequentes. Essas 
    verificações são essenciais em diferentes fases do ciclo de vida do elemento, uma vez que fatores como fabricação, transporte 
    e instalação podem alterar significativamente o diagrama de esforços e, consequentemente, o comportamento estrutural. As equações 
    (3.1) e (3.2) permitem calcular as tensões na base e no topo da seção transversal em qualquer fase da construção:
    """)

    st.latex(r"""
    \begin{align*}
        \sigma^{base} &= \frac{P_{Sd}}{A_c} + \frac{P_{Sd} \cdot e_p}{W_{base}} - \frac{M_{Sd}}{W_{base}} \tag{3.1} \\\\
        \sigma^{topo} &= \frac{P_{Sd}}{A_c} - \frac{P_{Sd} \cdot e_p}{W_{topo}} + \frac{M_{Sd}}{W_{topo}} \tag{3.2}
    \end{align*}
    """)

    st.write("""
    Onde:
    - $P_{Sd}$ é a força de protensão no instante $t$, considerando as perdas acumuladas até o período analisado;
    - $M_{Sd}$ é o momento fletor de cálculo para a combinação de ações considerada;
    - $A_c$ e $W$ representam, respectivamente, a área bruta da seção e o módulo resistente da seção transversal.

    Os valores das tensões limites para checagem foram definidos conforme a Tabela 3.1 e as equações (3.3) e (3.4):
    """)

    st.latex(r"""
    \begin{array}{|c|c|c|}
        \hline
        \textbf{Tipo checagem} & \textbf{Limite Compressão} & \textbf{Limite tração} \\
        \hline
        \text{Ato da protensão} & \text{ver equação (3.3)} & -1,20 \cdot f_{ctmj} \\
        \hline
        \text{Serviço} & 0,60 \cdot f_{ck} & -0,70 \cdot \kappa \cdot f_{ctm} \\
        \hline
    \end{array}
    """)

    st.write("*Tabela 3.1* Limitação de tensão conforme edição da Tabela xx da ABNT NBR 6118.")


    st.latex(r"""
    \begin{align*}
        \sigma_{c,lim} &=
            \begin{cases} 
            0{,}70 \cdot f_{ck}, & \text{para } f_{ck} \leq 50 \text{ MPa} \\
            0{,}70 \cdot \left[ 1 - \frac{f_{ck} - 50}{200} \right] f_{ck}, & \text{para } 50 < f_{ck} \leq 90 \text{ MPa} \tag{3.3}
            \end{cases} \\
             \\
        \kappa &=
        \begin{cases} 
        1,20 & \text{para seções T ou duplo T} \\
        1,30 & \text{para seções I ou T invertido} \\
        1,50 & \text{para seções retangulares} \tag{3.4}
        \end{cases}
             
    \end{align*}
    """)


    st.header("Verificação de flecha")

    st.markdown(r"""
        No caso da flecha o estudo também é similar ao concreto armado. No entanto, em peças protendidas, deve-se considerar a influência 
        da fluência do concreto na flecha total. A equação (3.5) expressa o cálculo da flecha para cargas uniformemente distribuídas 
        $\left(\delta_{g},\;\delta_{q}\;\text{ou}\;\delta_{g,ext}\right)$ aplicadas na peça, enquanto a equação (3.6) considera apenas 
        o efeito da protensão $\left(\delta_{p}\right)$.
        """)
    
    st.latex(r"""
    \begin{align*}
    \delta_{g \; \text{ou} \; q} &= \frac{5 \cdot q \cdot L^{4}}{384 \cdot E_{cs} \cdot I} \tag{3.5} \\\\
    \delta_{p} &= \frac{P \cdot e_{p} \cdot L^{2}}{8 \cdot E_{csj} \cdot I} \tag{3.6}
    \end{align*}
    """)

    st.markdown(r"""
    A flecha total deve ser avaliada tanto no ato da protensão $\left(\delta_{t, ato}\right)$ quanto em serviço $\left(\delta_{t, serv}\right)$, 
    sendo calculada conforme as equações (3.7) e (3.8):
    """)

    st.latex(r"""
    \begin{align*}
    \delta_{t, ato} &= -\delta_{p} + \delta_{g,pp} \tag{3.7} \\\\
    \delta_{t, serv} &= \phi_a \cdot \delta_{t, ato} + \phi_b \cdot \left(\delta_{g,ext} + \delta_{q}\right) \tag{3.8}
    \end{align*}
    """)

    st.markdown(r"""
    Onde:
    - $\phi_a$ e $\phi_b$ indicam os coeficientes de fluência para cada uma das etapas podendo variar de 1 a 3.

    Os limites de flecha são estabelecidos com base nas recomendações do item xx da ABNT NBR 9062 para o ato da protensão e item xx 
    da ABNT NBR 6118 para a aceitabilidade sensorial em serviço. Os valores são definidos pelas equações (3.9) e (3.10):
    """)

    st.latex(r"""
    \begin{align*}
    \delta_{lim,ato} &= \frac{L}{1000} \tag{3.9} \\\\
    \delta_{lim,serv} &= \frac{L}{250} \tag{3.10}
    \end{align*}
    """)

    st.header("Verificação da geometria")

    st.write("""
    Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy 
    text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has 
    survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was 
    popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop 
    publishing software like Aldus PageMaker including versions of Lorem Ipsum.
    """)
    st.write("")

if __name__ == "__main__":
    texto_01()