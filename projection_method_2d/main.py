import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Simulation Parameters
GRID_SIZE = 129
DOMAIN_LENGTH = 1.0
NUM_STEPS = 10000
DT = 0.001
# NU = 0.01  # Kinematic viscosity value  = 0.01 for re = 100
NU = 0.0025     # Kinematic viscosity  value = 0.025 for re = 400
RHO = 1.0        # Fluid density
TOP_LID_VELOCITY = 1.0
POISSON_ITER = 300


def run_simulation():
    dx = DOMAIN_LENGTH / (GRID_SIZE - 1)
    x_vals = np.linspace(0, DOMAIN_LENGTH, GRID_SIZE)
    y_vals = np.linspace(0, DOMAIN_LENGTH, GRID_SIZE)
    X, Y = np.meshgrid(x_vals, y_vals)

    u = np.zeros_like(X)
    v = np.zeros_like(X)
    p = np.zeros_like(X)

    def d_dx(f):        # using central difference scheme  
        res = np.zeros_like(f)
        res[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
        return res

    def d_dy(f):        # using central difference scheme
        res = np.zeros_like(f)
        res[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
        return res

    def laplacian(f):   # using central difference scheme
        res = np.zeros_like(f)
        res[1:-1, 1:-1] = (
            f[1:-1, 2:] + f[1:-1, :-2] +
            f[2:, 1:-1] + f[:-2, 1:-1] -
            4 * f[1:-1, 1:-1]
        ) / dx**2
        return res

    def pressure_solver(p, u, v):   # pressure poisson equation solver
        b = np.zeros_like(p)   # b is the rhs terms in pressure poisson equation i.e., b=  du/dx + dv/dy
        b[1:-1, 1:-1] = RHO * (
            1/DT * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                    (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx))
        )

        for _ in range(POISSON_ITER):       # here i am using a jacobi iteration, it was difficult to implement sor in vecotrized manner
            p[1:-1, 1:-1] = (
                (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - dx**2 * b[1:-1, 1:-1]) / 4
            )
            p[:, -1] = p[:, -2]  # Right boundary
            p[:, 0] = p[:, 1]    # Left boundary
            p[0, :] = p[1, :]    # Bottom boundary
            # p[-1, :] = 0         # Top boundary - drichlet
            p[-1, :] = p[-2, :]     # neuman boundary condition
        return p
        



    velocity_history = []

    # i found that running for a sufficiently high number of iternations gave better result than using convergence criteria somehow
    for step in tqdm(range(NUM_STEPS)):
        

        u_star = u.copy()
        v_star = v.copy()

        u_star[1:-1, 1:-1] += DT * (
            - u[1:-1, 1:-1] * d_dx(u)[1:-1, 1:-1] - v[1:-1, 1:-1] * d_dy(u)[1:-1, 1:-1] +
            NU * laplacian(u)[1:-1, 1:-1]
        )

        v_star[1:-1, 1:-1] += DT * (
            - u[1:-1, 1:-1] * d_dx(v)[1:-1, 1:-1] - v[1:-1, 1:-1] * d_dy(v)[1:-1, 1:-1] +
            NU * laplacian(v)[1:-1, 1:-1]
        )

        p = pressure_solver(p, u_star, v_star)

        # now enforcing incompressible flow property 

        u_star[1:-1, 1:-1] -= DT / RHO * d_dx(p)[1:-1, 1:-1]
        v_star[1:-1, 1:-1] -= DT / RHO * d_dy(p)[1:-1, 1:-1]

        # Boundary conditions
        u_star[0, :] = u_star[-1, :] = u_star[:, 0] = u_star[:, -1] = 0
        v_star[0, :] = v_star[-1, :] = v_star[:, 0] = v_star[:, -1] = 0
        u_star[-1, :] = TOP_LID_VELOCITY

        u = u_star
        v = v_star

        if step % 100 == 0:
            velocity_magnitude = np.sqrt(u**2 + v**2)
            velocity_history.append(velocity_magnitude.copy())

    def divergence(u, v):

        # this was the best i could do to minimize divergence at upper boundary. But some divergence still remains at corners
        divergence  = np.zeros_like(u)
        divergence[1:-1, 1:-1] = d_dx(u)[1:-1, 1:-1] + d_dy(v)[1:-1, 1:-1]


        # bottom boundary
        divergence[0, 1:-1] = (
            (u[0, 2:] - u[0, 1:-1]) / dx +
            (v[1, 1:-1] - v[0, 1:-1]) / dx
        )

        # top boundary
        divergence[-1, 1:-1] = (
            (u[-1, 1:-1] - u[-1, :-2]) / dx +
            (v[-1, 1:-1] - v[-2, 1:-1]) / dx
        )

        # left boundary
        divergence[1:-1, 0] = (
            (u[1:-1, 1] - u[1:-1, 0]) / dx +
            (v[2:, 0] - v[:-2, 0]) / (2 * dx)
        )

        # right boundary
        divergence[1:-1, -1] = (
            (u[1:-1, -1] - u[1:-1, -2]) / dx +
            (v[2:, -1] - v[:-2, -1]) / (2 * dx)
        )

        # used one sided differences in corners
        divergence[0, 0] = ((u[0, 1] - u[0, 0]) / dx + (v[1, 0] - v[0, 0]) / dx)
        divergence[0, -1] = ((u[0, -1] - u[0, -2]) / dx + (v[1, -1] - v[0, -1]) / dx)
        divergence[-1, 0] = ((u[-1, 1] - u[-1, 0]) / dx + (v[-1, 0] - v[-2, 0]) / dx)
        divergence[-1, -1] = ((u[-1, -1] - u[-1, -2]) / dx + (v[-1, -1] - v[-2, -1]) / dx)

        return divergence

    divergence_val = divergence(u,v)

    return X, Y, u, v, p, velocity_history, divergence_val


if __name__ == '__main__':

    X, Y, u, v, p, velocity_log, divergence = run_simulation()

    df1 = pd.DataFrame(u[:, 64])
    # Save to CSV
    df1.to_csv("cfd_re400_n.csv")
    print("CSV file saved as cfd_re400_n.csv")

    np.save('velocity_log.npy', velocity_log)
    print('saved velocity_log.npy')
    

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # this is the velocity stream plot
    ax1 = axes[0]
    ax1.streamplot(X, Y, u, v, density=1.2)
    ax1.set_title('2D Lid-Driven Cavity Flow — Velocity Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # this is a heatmap for divergence magnitude
    ax2 = axes[1]
    im = ax2.imshow(divergence, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    ax2.set_title("Divergence Magnitude Heatmap (Cell View)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(False)

    # Colorbar for divergence on the right plot only
    fig.colorbar(im, ax=ax2, label='Divergence magnitude')

    plt.tight_layout()
    plt.show()

   

    
