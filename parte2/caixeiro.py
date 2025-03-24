import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from typing import List, Tuple

# Configurações do algoritmo
NUM_CIDADES = 20
GERACOES = 100
POPULACAO_SIZE = 100
MUTACAO_CHANCE = 0.1
ELITISMO = 2

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
        random.sample(range(NUM_CIDADES), NUM_CIDADES) for _ in range(POPULACAO_SIZE)
    ]


def selecao_torneio(populacao: List[List[int]]) -> List[int]:
    competidores = random.sample(populacao, 5)
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
    if random.random() < MUTACAO_CHANCE:
        i, j = random.sample(range(NUM_CIDADES), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]


# Algoritmo genético principal
def algoritmo_genetico():
    global historico_rotas, historico_distancias
    historico_rotas = []
    historico_distancias = []
    populacao = gerar_populacao()
    melhor_solucao = None
    melhor_fitness = float("-inf")

    for geracao in range(GERACOES):
        populacao = sorted(populacao, key=calcular_fitness, reverse=True)
        nova_populacao = populacao[:ELITISMO]

        while len(nova_populacao) < POPULACAO_SIZE:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho1, filho2 = crossover(pai1, pai2)
            mutacao(filho1)
            mutacao(filho2)
            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:POPULACAO_SIZE]
        melhor_individuo = max(populacao, key=calcular_fitness)
        melhor_valor = calcular_fitness(melhor_individuo)

        if melhor_valor > melhor_fitness:
            melhor_solucao = melhor_individuo.copy()
            melhor_fitness = melhor_valor

        historico_rotas.append(melhor_solucao.copy())
        historico_distancias.append(-melhor_valor)

    return melhor_solucao, historico_rotas, historico_distancias


# Executar o algoritmo e armazenar histórico
solucao, historico_rotas, historico_distancias = algoritmo_genetico()

# Configurar a interface gráfica principal
fig1, ax1 = plt.subplots(figsize=(10, 7), num="TSP Route Evolution")
plt.subplots_adjust(bottom=0.25)

# Plot city markers
ax1.scatter(CIDADES[:, 0], CIDADES[:, 1], c="blue", s=100, zorder=3)

# Draw all possible edges as background
all_edges = []
for i in range(NUM_CIDADES):
    for j in range(i + 1, NUM_CIDADES):
        all_edges.append([CIDADES[i], CIDADES[j]])

bg_lines = LineCollection(
    all_edges, colors="lightgray", alpha=0.15, linewidths=1, zorder=1
)
ax1.add_collection(bg_lines)

# Initialize active edges collection
active_edges = LineCollection([], colors="red", linewidths=2, alpha=0.8, zorder=2)
ax1.add_collection(active_edges)

titulo = ax1.set_title("Geração: 0 | Distância: 0.00")
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_axis_off()

# Configurar gráfico de evolução
fig2, ax2 = plt.subplots(figsize=(10, 4), num="Distance Evolution")
ax2.set_title("Evolução da Distância")
ax2.set_xlabel("Geração")
ax2.set_ylabel("Distância")
ax2.grid(True)

# Plot baseline
generations = np.arange(1, len(historico_distancias) + 1)
(line,) = ax2.plot(generations, historico_distancias, lw=2)
current_marker = ax2.axvline(x=0, color="r", linestyle="--", alpha=0.7)
ax2.set_xlim(0, len(historico_distancias))
ax2.set_ylim(min(historico_distancias) * 0.95, max(historico_distancias) * 1.05)

# Criar controles na janela principal
# Criar controles na janela principal
ax_slider = fig1.add_axes((0.2, 0.1, 0.6, 0.03))
generation_slider = Slider(
    ax=ax_slider,
    label="Geração",
    valmin=0,
    valmax=len(historico_rotas) - 1,
    valinit=0,
    valstep=1,
)

ax_play = fig1.add_axes((0.7, 0.05, 0.1, 0.04))
play_button = Button(ax_play, "▶")
ax_pause = fig1.add_axes((0.81, 0.05, 0.1, 0.04))
pause_button = Button(ax_pause, "⏸")
# Variáveis de controle
is_playing = False
current_frame = 0


# Funções de atualização
def update_slider(val):
    global current_frame
    current_frame = int(generation_slider.val)
    update_plot(current_frame)


def update_plot(frame):
    # Update TSP route
    route = historico_rotas[frame]
    segments = []
    for i in range(len(route)):
        start = CIDADES[route[i]]
        end = CIDADES[route[(i + 1) % len(route)]]
        segments.append([start, end])
    active_edges.set_segments(segments)
    titulo.set_text(
        f"Geração: {frame + 1} | Distância: {historico_distancias[frame]:.2f}"
    )

    # Update evolution chart
    current_marker.set_xdata([frame + 1, frame + 1])
    fig2.canvas.draw_idle()

    # Redraw both figures
    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()


def play(event):
    global is_playing
    is_playing = True
    animate()


def pause(event):
    global is_playing
    is_playing = False


def animate():
    global current_frame, is_playing
    if is_playing and current_frame < len(historico_rotas) - 1:
        current_frame += 1
        generation_slider.set_val(current_frame)
        plt.pause(0.1)
        animate()


# Conectar eventos
generation_slider.on_changed(update_slider)
play_button.on_clicked(play)
pause_button.on_clicked(pause)

# Mostrar resultado final
print("Melhor rota encontrada:", solucao)
print("Menor distância encontrada:", historico_distancias[-1])

# Inicializar com a primeira geração
update_plot(0)

# Mostrar ambas as janelas
plt.show()
