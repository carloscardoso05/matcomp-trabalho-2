import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# ----- Lévy function definition (2D) -----
def levy_2d(x, y):
    """
    x, y: scalar or np.array
    Returns the Lévy function value at (x, y).
    """
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)
    return term1 + term2 + term3

# ----- PSO implementation for Lévy (2D) that stores iterations -----
def pso_levy(num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5, bound=(-10, 10)):
    lower, upper = bound
    # Random initial positions and velocities
    particles_pos = np.random.uniform(lower, upper, (num_particles, 2))
    particles_vel = np.random.uniform(-1, 1, (num_particles, 2))
    
    fitness = np.array([levy_2d(p[0], p[1]) for p in particles_pos])
    pbest_pos = particles_pos.copy()
    pbest_val = fitness.copy()
    
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    
    all_positions = []  # store positions each iteration
    
    for _ in range(max_iter):
        all_positions.append(particles_pos.copy())
        for i in range(num_particles):
            r1 = np.random.rand()
            r2 = np.random.rand()
            particles_vel[i] = (
                w * particles_vel[i]
                + c1 * r1 * (pbest_pos[i] - particles_pos[i])
                + c2 * r2 * (gbest_pos - particles_pos[i])
            )
            particles_pos[i] += particles_vel[i]
            particles_pos[i] = np.clip(particles_pos[i], lower, upper)
            val = levy_2d(particles_pos[i][0], particles_pos[i][1])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = particles_pos[i].copy()
        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
    all_positions.append(particles_pos.copy())
    return all_positions, gbest_pos, gbest_val

# ----- Create a 3D mesh for the Lévy function -----
def create_levy_mesh(xmin=-10, xmax=10, ymin=-10, ymax=10, points=200):
    x_vals = np.linspace(xmin, xmax, points)
    y_vals = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = levy_2d(X, Y)
    return X, Y, Z

# ----- 3D Animation for PSO on the Lévy function -----
def animate_pso_levy_3d():
    all_positions, gbest_pos, gbest_val = pso_levy(
        num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5, bound=(-10, 10)
    )
    X, Y, Z = create_levy_mesh(-10, 10, -10, 10, points=200)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    
    # Initial scatter: start with an empty scatter (we'll update its data)
    scatter = ax.scatter([], [], [], color='red', s=40)
    
    # For the title, since 3D doesn't support blitting, we'll update the figure title directly
    def update(frame):
        positions = all_positions[frame]
        x = positions[:, 0]
        y = positions[:, 1]
        z = levy_2d(x, y)
        # Update the 3D scatter's positions
        scatter._offsets3d = (x, y, z)
        ax.set_title(f'PSO on Lévy 3D - Iteration {frame + 1}/{len(all_positions)}')
        return scatter,
    
    ani = FuncAnimation(fig, update, frames=len(all_positions), interval=200, repeat=False)
    plt.show()

# Run the 3D animation
animate_pso_levy_3d()
