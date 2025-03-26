import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Definição da função de Lévy (mantida igual)
def levy_2d(x, y):
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)
    return term1 + term2 + term3

# Modificação do PSO para armazenar histórico de fitness
def enxame_levy(num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5, bound=(-10, 10)):
    lower, upper = bound
    
    particles_pos = np.random.uniform(lower, upper, (num_particles, 2))
    particles_vel = np.random.uniform(-1, 1, (num_particles, 2))
    
    fitness = np.array([levy_2d(p[0], p[1]) for p in particles_pos])
    pbest_pos = particles_pos.copy()
    pbest_val = fitness.copy()
    
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    
    all_positions = []
    best_fit_history = [gbest_val]  # Histórico do melhor fitness
    worst_fit_history = [np.max(pbest_val)]  # Histórico do pior fitness
    
    for _ in range(max_iter):
        all_positions.append(particles_pos.copy())
        
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            particles_vel[i] = w * particles_vel[i] + c1 * r1 * (pbest_pos[i] - particles_pos[i]) + c2 * r2 * (gbest_pos - particles_pos[i])
            particles_pos[i] = np.clip(particles_pos[i] + particles_vel[i], lower, upper)
            
            # Avaliar nova posição
            new_fitness = levy_2d(particles_pos[i][0], particles_pos[i][1])
            if new_fitness < pbest_val[i]:
                pbest_val[i] = new_fitness
                pbest_pos[i] = particles_pos[i].copy()
                
        # Atualizar histórico
        current_best = np.min(pbest_val)
        current_worst = np.max(pbest_val)
        best_fit_history.append(current_best)
        worst_fit_history.append(current_worst)
        
        if current_best < gbest_val:
            gbest_val = current_best
            gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()
    
    all_positions.append(particles_pos.copy())
    return all_positions, gbest_pos, gbest_val, best_fit_history, worst_fit_history

# Função de animação atualizada com gráfico de convergência
def animate_enxame_levy_3d():
    all_positions, gbest_pos, gbest_val, best_fit, worst_fit = enxame_levy()
    X, Y, Z = create_levy_mesh(-10, 10, -10, 10)
    
    # Create two separate figures
    fig1 = plt.figure(figsize=(8, 6))
    fig2 = plt.figure(figsize=(8, 6))
    
    # 3D plot setup
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    scatter = ax1.scatter([], [], [], c='red', s=40)
    ax1.set_title('Exploração do Enxame na Função de Lévy 3D')
    
    # Static convergence plot setup
    ax2 = fig2.add_subplot(111)
    iterations = np.arange(len(best_fit))
    ax2.plot(iterations, best_fit, 'b-', label='Melhor Fitness')
    ax2.plot(iterations, worst_fit, 'r--', label='Pior Fitness')
    ax2.legend()
    ax2.set_title('Convergência do Fitness')
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Valor do Fitness')
    ax2.grid(True)
    
    def update(frame):
        # Update only 3D plot
        positions = all_positions[frame]
        x, y = positions[:,0], positions[:,1]
        z = levy_2d(x, y)
        scatter._offsets3d = (x, y, z)
        ax1.set_title(f'Iteração {frame + 1}/{len(all_positions)}')
        return scatter,
    
    # Create animation only for 3D plot without repeating
    ani1 = FuncAnimation(fig1, update, frames=len(all_positions), interval=200, blit=True, repeat=False)
    
    # Show both figures with non-blocking mode
    plt.show(block=False)
    
    # Keep the script running until animation is complete
    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except KeyboardInterrupt:
        plt.close('all')

# Mantemos a função de criação da malha igual
def create_levy_mesh(xmin=-10, xmax=10, ymin=-10, ymax=10, points=200):
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x, y)
    Z = levy_2d(X, Y)
    return X, Y, Z

animate_enxame_levy_3d()