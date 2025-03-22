import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from scipy.optimize import differential_evolution

# ----- Ackley Function Definition -----
def ackley_2d(x, y, a=20, b=0.2, c=2*np.pi):
    """
    Computes the Ackley function value at given x and y.
    Used for meshgrid evaluations.
    """
    term1 = -a * np.exp(-b * np.sqrt(0.5*(x**2 + y**2)))
    term2 = -np.exp(0.5*(np.cos(c*x) + np.cos(c*y)))
    return term1 + term2 + a + np.e

# Wrapper for differential evolution
def ackley_vector(v, a=20, b=0.2, c=2*np.pi):
    """
    Wrapper function for differential evolution.
    Accepts a vector v = [x, y] and returns the Ackley function value.
    """
    x, y = v
    return ackley_2d(x, y, a, b, c)

# ----- Create a Mesh for the Ackley Function -----
def create_ackley_mesh(xmin=-5, xmax=5, ymin=-5, ymax=5, points=200):
    x_vals = np.linspace(xmin, xmax, points)
    y_vals = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = ackley_2d(X, Y)
    return X, Y, Z

# ----- Global List to Store Populations per Iteration -----
all_populations_ackley = []

def callback_de_ackley(population, convergence):
    """
    Callback function for DE. Stores the current population at each iteration.
    """
    all_populations_ackley.append(population.copy())

# ----- Run Differential Evolution and Animate in 3D -----
def animate_de_ackley_3d():
    global all_populations_ackley
    all_populations_ackley = []  # Reset the list

    # Run Differential Evolution on the Ackley function using the wrapper
    bounds = [(-5, 5), (-5, 5)]
    result = differential_evolution(
        ackley_vector,  # use the vectorized function
        bounds=bounds,
        strategy='best1bin',
        maxiter=30,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=callback_de_ackley,
        seed=42
    )
    print("DE Ackley => Best pos:", result.x, "Best val:", result.fun)

    # Create a mesh grid for the Ackley function surface
    X, Y, Z = create_ackley_mesh(-5, 5, -5, 5, points=200)

    # Setup the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface of the Ackley function
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    # Initialize an empty scatter plot for the DE population
    scatter = ax.scatter([], [], [], color='red', s=40)

    # Update function for the animation
    def update(frame):
        pop = all_populations_ackley[frame]
        x = pop[:, 0]
        y = pop[:, 1]
        z = ackley_2d(x, y)
        # Update the 3D scatter plot (using _offsets3d)
        scatter._offsets3d = (x, y, z)
        ax.set_title(f'DE on Ackley 3D - Iteration {frame+1}/{len(all_populations_ackley)}')
        return scatter,

    # Create the animation (note: 3D animations don't support blitting)
    ani = FuncAnimation(fig, update, frames=len(all_populations_ackley), interval=300, repeat=False)
    plt.show()

# Run the 3D animation for the Ackley function using Differential Evolution
animate_de_ackley_3d()
