import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def griewank_2d(x, y):
    sum_term = (x**2)/4000 + (y**2)/4000
    prod_term = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return sum_term - prod_term + 1

def griewank_vector(v):
    x, y = v
    return griewank_2d(x, y)

def create_griewank_mesh(xmin=-10, xmax=10, ymin=-10, ymax=10, points=200):
    x_vals = np.linspace(xmin, xmax, points)
    y_vals = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = griewank_2d(X, Y)
    return X, Y, Z

def pso_griewank(num_particles=30, max_iter=50, bound=(-10, 10), w=0.7, c1=1.5, c2=1.5):
    lower, upper = bound
    dim = 2
    
    particles = np.random.uniform(lower, upper, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    personal_best = particles.copy()
    personal_best_values = np.array([griewank_vector(p) for p in particles])
    global_best_idx = np.argmin(personal_best_values)
    global_best = personal_best[global_best_idx]
    
    all_positions = [particles.copy()]
    best_fit_history = [np.min(personal_best_values)]
    worst_fit_history = [np.max(personal_best_values)]
    
    for _ in range(max_iter):
        r1, r2 = np.random.rand(num_particles, dim), np.random.rand(num_particles, dim)
        velocities = (w * velocities + c1 * r1 * (personal_best - particles) + c2 * r2 * (global_best - particles))
        particles = np.clip(particles + velocities, lower, upper)
        
        values = np.array([griewank_vector(p) for p in particles])
        update_mask = values < personal_best_values
        personal_best[update_mask] = particles[update_mask]
        personal_best_values[update_mask] = values[update_mask]
        
        new_global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[new_global_best_idx]
        
        all_positions.append(particles.copy())
        best_fit_history.append(np.min(personal_best_values))
        worst_fit_history.append(np.max(personal_best_values))
    
    return all_positions, global_best, np.min(personal_best_values), best_fit_history, worst_fit_history

def animate_pso_griewank_3d():
    all_positions, best_sol, best_val, best_fit, worst_fit = pso_griewank()
    print(f"PSO => Best pos: {best_sol}, Best val: {best_val}")
    
    X, Y, Z = create_griewank_mesh(-10, 10, -10, 10, points=200)
    
    fig1 = plt.figure(figsize=(8, 6))
    fig2 = plt.figure(figsize=(8, 6))
    
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    scatter = ax1.scatter([], [], [], color='red', s=40)
    
    ax2 = fig2.add_subplot(111)
    iterations = np.arange(len(best_fit))
    ax2.plot(iterations, best_fit, 'b-', label='Best Fitness')
    ax2.plot(iterations, worst_fit, 'r--', label='Worst Fitness')
    ax2.legend()
    ax2.set_title('Fitness Convergence')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness Value')
    ax2.grid(True)
    
    def update(frame):
        pop = all_positions[frame]
        x = pop[:, 0]
        y = pop[:, 1]
        z = griewank_2d(x, y)
        scatter._offsets3d = (x, y, z)
        ax1.set_title(f'PSO on Griewank (3D) - Iter {frame+1}/{len(all_positions)}')
        return (scatter,)
    
    ani1 = FuncAnimation(fig1, update, frames=len(all_positions), interval=300, repeat=False)
    plt.show(block=False)
    
    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except KeyboardInterrupt:
        plt.close('all')

if __name__ == "__main__":
    animate_pso_griewank_3d()
