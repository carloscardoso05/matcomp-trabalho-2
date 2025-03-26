import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from typing import List, Tuple

# Configurações do algoritmo
NUM_CIDADES = 20
GERACOES = 100
TAMANHO_POPULACAO = 300
CHANCE_MUTACAO = 0.2
ELITISMO = int(TAMANHO_POPULACAO * 0.05) # 5% da população
NUM_COMPETIDORES = 7

# Gerar coordenadas aleatórias para as cidades
CIDADES = np.random.rand(NUM_CIDADES, 2)


def calcular_distancia(cidade1, cidade2):
    return np.linalg.norm(cidade1 - cidade2)


# Matriz de distâncias
DISTANCIAS = np.array(
    [
        [calcular_distancia(CIDADES[i], CIDADES[j]) for j in range(NUM_CIDADES)]
        for i in range(NUM_CIDADES)
    ]
)


def calcular_fitness(individuo: List[int]) -> float:
    total = sum(
        DISTANCIAS[individuo[i], individuo[i + 1]] for i in range(NUM_CIDADES - 1)
    )
    total += DISTANCIAS[individuo[-1], individuo[0]]
    return -total


def gerar_populacao() -> List[List[int]]:
    return [
        random.sample(range(NUM_CIDADES), NUM_CIDADES) for _ in range(TAMANHO_POPULACAO)
    ]


def selecao_torneio(populacao: List[List[int]]) -> List[int]:
    competidores = random.sample(populacao, NUM_COMPETIDORES)
    return max(competidores, key=calcular_fitness)


def crossover(pai1: List[int], pai2: List[int]) -> Tuple[List[int], List[int]]:
    p1, p2 = sorted(random.sample(range(NUM_CIDADES), 2))
    filho1 = [-1] * NUM_CIDADES
    filho2 = [-1] * NUM_CIDADES

    filho1[p1:p2] = pai1[p1:p2]
    filho2[p1:p2] = pai2[p1:p2]

    def preencher_filho(filho, pai):
        pos = p2
        for gene in pai:
            if gene not in filho:
                if pos >= NUM_CIDADES:
                    pos = 0
                filho[pos] = gene
                pos += 1

    preencher_filho(filho1, pai2)
    preencher_filho(filho2, pai1)

    return filho1, filho2


def mutacao(individuo: List[int]):
    if random.random() < CHANCE_MUTACAO:
        i, j = random.sample(range(NUM_CIDADES), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]


# Algoritmo genético principal
def algoritmo_genetico():
    global historico_rotas, historico_distancias, historico_pior_distancias
    historico_rotas = []
    historico_distancias = []
    historico_pior_distancias = []
    populacao = gerar_populacao()
    melhor_solucao = None
    melhor_fitness = float("-inf")

    for geracao in range(GERACOES):
        populacao = sorted(populacao, key=calcular_fitness, reverse=True)
        nova_populacao = populacao[:ELITISMO]

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho1, filho2 = crossover(pai1, pai2)
            mutacao(filho1)
            mutacao(filho2)
            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:TAMANHO_POPULACAO]
        melhor_individuo = max(populacao, key=calcular_fitness)
        pior_individuo = min(populacao, key=calcular_fitness)
        melhor_valor = calcular_fitness(melhor_individuo)
        pior_valor = calcular_fitness(pior_individuo)

        if melhor_valor > melhor_fitness:
            melhor_solucao = melhor_individuo.copy()
            melhor_fitness = melhor_valor

        historico_rotas.append(melhor_solucao.copy())
        historico_distancias.append(-melhor_valor)
        historico_pior_distancias.append(-pior_valor)

    return melhor_solucao, historico_rotas, historico_distancias, historico_pior_distancias


# Executar o algoritmo e armazenar histórico
solucao, historico_rotas, historico_distancias, historico_pior_distancias = algoritmo_genetico()

# Configurar a interface gráfica principal
fig1, ax1 = plt.subplots(figsize=(10, 7), num="Evolução da Rota TSP")
plt.subplots_adjust(bottom=0.25)

# Plotar marcadores das cidades
ax1.scatter(CIDADES[:, 0], CIDADES[:, 1], c="blue", s=100, zorder=3)

# Desenhar todas as arestas possíveis como fundo
todas_arestas = []
for i in range(NUM_CIDADES):
    for j in range(i + 1, NUM_CIDADES):
        todas_arestas.append([CIDADES[i], CIDADES[j]])

linhas_fundo = LineCollection(
    todas_arestas, colors="lightgray", alpha=0.15, linewidths=1, zorder=1
)
ax1.add_collection(linhas_fundo)

# Inicializar coleção de arestas ativas
arestas_ativas = LineCollection([], colors="red", linewidths=2, alpha=0.8, zorder=2)
ax1.add_collection(arestas_ativas)

titulo = ax1.set_title("Geração: 0 | Distância: 0.00")
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_axis_off()

# Configurar gráfico de evolução
fig2, ax2 = plt.subplots(figsize=(12, 6), num="Evolução da Distância", facecolor='white')
ax2.set_facecolor('white')

# Personalizar o gráfico
ax2.set_title("Evolução da Distância", fontsize=14, pad=15)
ax2.set_xlabel("Geração", fontsize=12)
ax2.set_ylabel("Distância", fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Configurar grade
ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
ax2.set_axisbelow(True)  # Colocar grade abaixo dos gráficos

# Remover bordas superior e direita
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#333333')
ax2.spines['bottom'].set_color('#333333')

# Plotar linha base com melhores e piores distâncias com estilo melhorado
geracoes = np.arange(1, len(historico_distancias) + 1)
linha_melhor, = ax2.plot(geracoes, historico_distancias, 
                      color='#1f77b4',  # Cor azul agradável
                      lw=2.5, 
                      label='Melhor Distância',
                      solid_capstyle='round')
linha_pior, = ax2.plot(geracoes, historico_pior_distancias, 
                       color='#d62728',  # Cor vermelha agradável
                       lw=2, 
                       label='Pior Distância',
                       linestyle='--',
                       alpha=0.8,
                       dash_capstyle='round')

# Melhorar legenda
legenda = ax2.legend(loc='upper right', 
                   frameon=True,
                   framealpha=0.95,
                   edgecolor='#333333',
                   fontsize=10)
legenda.get_frame().set_facecolor('white')

# Adicionar marcador de geração com melhor estilo
marcador_atual = ax2.axvline(x=0, 
                            color='#2ca02c',  # Cor verde agradável
                            linestyle='--', 
                            alpha=0.6,
                            lw=1.5)

# Definir melhores limites dos eixos
ax2.set_xlim(0, len(historico_distancias))
y_min = min(historico_distancias) * 0.95
y_max = max(historico_pior_distancias) * 1.05
ax2.set_ylim(y_min, y_max)

# Adicionar ticks menores
ax2.minorticks_on()
ax2.tick_params(which='minor', length=2)
ax2.tick_params(which='major', length=4)

# Ajustar layout
plt.tight_layout()

# Criar controles na janela principal
ax_slider = fig1.add_axes((0.2, 0.1, 0.6, 0.03))
slider_geracao = Slider(
    ax=ax_slider,
    label="Geração",
    valmin=0,
    valmax=len(historico_rotas) - 1,
    valinit=0,
    valstep=1,
)

ax_play = fig1.add_axes((0.7, 0.05, 0.1, 0.04))
botao_play = Button(ax_play, "▶")
ax_pause = fig1.add_axes((0.81, 0.05, 0.1, 0.04))
botao_pause = Button(ax_pause, "⏸")

# Variáveis de controle
reproduzindo = False
frame_atual = 0


# Funções de atualização
def atualizar_slider(val):
    global frame_atual
    frame_atual = int(slider_geracao.val)
    atualizar_grafico(frame_atual)


def atualizar_grafico(frame):
    # Atualizar rota TSP
    rota = historico_rotas[frame]
    segmentos = []
    for i in range(len(rota)):
        inicio = CIDADES[rota[i]]
        fim = CIDADES[rota[(i + 1) % len(rota)]]
        segmentos.append([inicio, fim])
    arestas_ativas.set_segments(segmentos)
    titulo.set_text(
        f"Geração: {frame + 1} | Melhor: {historico_distancias[frame]:.2f} | Pior: {historico_pior_distancias[frame]:.2f}"
    )

    # Atualizar gráfico de evolução
    marcador_atual.set_xdata([frame + 1, frame + 1])
    fig2.canvas.draw_idle()

    # Redesenhar ambas as figuras
    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()


def play(event):
    global reproduzindo
    reproduzindo = True
    animar()


def pause(event):
    global reproduzindo
    reproduzindo = False


def animar():
    global frame_atual, reproduzindo
    if reproduzindo and frame_atual < len(historico_rotas) - 1:
        frame_atual += 1
        slider_geracao.set_val(frame_atual)
        plt.pause(0.1)
        animar()


# Conectar eventos
slider_geracao.on_changed(atualizar_slider)
botao_play.on_clicked(play)
botao_pause.on_clicked(pause)

# Mostrar resultado final
print("Melhor rota encontrada:", solucao)
print("Menor distância encontrada:", historico_distancias[-1])
print("Maior distância encontrada:", historico_pior_distancias[-1])

# Inicializar com a primeira geração
atualizar_grafico(0)

# Mostrar ambas as janelas
plt.show()
