import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ===========================
#   1) Griewank Definition
# ===========================
def griewank_2d(x, y):
    """
    Computes the 2D Griewank function at (x, y).
    """
    sum_term = (x**2)/4000 + (y**2)/4000
    prod_term = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return sum_term - prod_term + 1

# Vector version
def griewank_vector(v):
    x, y = v
    return griewank_2d(x, y)

# ===================================
#   2) Create Mesh for 3D Plotting
# ===================================
def create_griewank_mesh(xmin=-10, xmax=10, ymin=-10, ymax=10, points=200):
    x_vals = np.linspace(xmin, xmax, points)
    y_vals = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = griewank_2d(X, Y)
    return X, Y, Z

# =====================================
#   3) Differential Evolution (DE)
# =====================================
def differential_evolution_griewank(
    num_individuals=30,
    max_iter=50,
    bound=(-10, 10),
    F=0.8,  # Mutation factor
    CR=0.9  # Crossover probability
):
    """
    Runs Differential Evolution (DE) on the 2D Griewank function.
    Returns:
      - all_positions: List of arrays (num_individuals x 2) for each iteration
      - best_solution: (x, y) of best found solution
      - best_value: Griewank value at best_solution
    """
    lower, upper = bound
    dim = 2  # Since we are in 2D

    # Initialize population randomly
    population = np.random.uniform(lower, upper, (num_individuals, dim))

    # Evaluate initial fitness
    fitness = np.array([griewank_vector(ind) for ind in population])

    # Store all positions per iteration
    all_positions = [population.copy()]

    for iteration in range(max_iter):
        new_population = np.copy(population)
        
        for i in range(num_individuals):
            # Mutation: Select three random individuals
            idxs = [idx for idx in range(num_individuals) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Compute mutant vector
            mutant = np.clip(a + F * (b - c), lower, upper)
            
            # Crossover: Create a trial vector
            crossover_mask = np.random.rand(dim) < CR
            if not np.any(crossover_mask):  # Ensure at least one dimension changes
                crossover_mask[np.random.randint(0, dim)] = True
            
            trial = np.where(crossover_mask, mutant, population[i])

            # Selection: Choose the better one
            trial_fitness = griewank_vector(trial)
            if trial_fitness < fitness[i]:  # Minimize the function
                new_population[i] = trial
                fitness[i] = trial_fitness

        # Update population
        population = new_population.copy()
        all_positions.append(population.copy())

    # Get the best solution found
    best_idx = np.argmin(fitness)
    best_sol = population[best_idx]
    best_val = fitness[best_idx]

    return all_positions, best_sol, best_val

# ====================================
#   4) 3D Animation for Visualization
# ====================================
def animate_de_griewank_3d():
    # 1) Run Differential Evolution
    all_positions, best_sol, best_val = differential_evolution_griewank(
        num_individuals=30, 
        max_iter=50, 
        bound=(-10,10),
        F=0.8,
        CR=0.9
    )
    print(f"Differential Evolution => Best pos: {best_sol}, Best val: {best_val}")

    # 2) Create 3D surface for visualization
    X, Y, Z = create_griewank_mesh(-10,10, -10,10, points=200)

    # 3) Set up the figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Griewank surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')

    # Initialize scatter for population
    scatter = ax.scatter([], [], [], color='red', s=40)

    # 4) Update function for animation
    def update(frame):
        pop = all_positions[frame]
        x = pop[:, 0]
        y = pop[:, 1]
        z = griewank_2d(x, y)

        # Update the 3D scatter
        scatter._offsets3d = (x, y, z)
        ax.set_title(f'DE on Griewank (3D) - Iter {frame+1}/{len(all_positions)}')
        return (scatter,)

    # 5) Animate
    ani = FuncAnimation(
        fig, update, 
        frames=len(all_positions), 
        interval=300, 
        repeat=False
    )
    plt.show()

# Run the animation
if __name__ == "__main__":
    animate_de_griewank_3d()
