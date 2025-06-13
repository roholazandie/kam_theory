import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def resonant_torus_motion(p: int, q: int, n_points: int = 8000):
    ω1, ω2 = float(p), float(q)
    T = 2 * np.pi
    t = np.linspace(0.0, T, n_points, endpoint=True)
    θ1 = (ω1 * t) % (2 * np.pi)
    θ2 = (ω2 * t) % (2 * np.pi)
    return θ1, θ2

def embed_torus(θ1, θ2, R=3.0, r=1.0):
    x = (R + r*np.cos(θ1)) * np.cos(θ2)
    y = (R + r*np.cos(θ1)) * np.sin(θ2)
    z =  r * np.sin(θ1)
    return x, y, z

def plot_resonant_torus_3d(p=2, q=5, n_points=8000, R=3.0, r=1.0,
                           torus_mesh_res=60, surface_alpha=0.25):
    θ1, θ2 = resonant_torus_motion(p, q, n_points)
    x, y, z = embed_torus(θ1, θ2, R, r)

    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # ----- draw the torus surface in semi-transparent blue -----
    u = np.linspace(0, 2*np.pi, torus_mesh_res)
    v = np.linspace(0, 2*np.pi, torus_mesh_res)
    U, V = np.meshgrid(u, v)
    X = (R + r*np.cos(U)) * np.cos(V)
    Y = (R + r*np.cos(U)) * np.sin(V)
    Z =  r * np.sin(U)
    ax.plot_surface(X, Y, Z, color='blue', alpha=surface_alpha,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)

    # ----- overlay the resonant trajectory -----
    ax.plot(x, y, z, color='orange', linewidth=1.6)

    ax.set_title(rf"Resonant orbit {p}/{q} on $T^2$", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # keep equal aspect
    L = R + r
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)

    plt.tight_layout()
    plt.show()

# --- Example:  ω₁/ω₂ = 2/5 ---------------------------------------
plot_resonant_torus_3d(p=10, q=1)
