import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

def resonant_torus_motion(p: int, q: int, n_points: int = 400):
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

def create_resonant_torus_animation(p=2, q=5, n_frames=400, R=3.0, r=1.0,
                                    torus_mesh_res=60, surface_alpha=0.25,
                                    fname="resonant_torus.mp4"):
    θ1, θ2 = resonant_torus_motion(p, q, n_points=n_frames)
    x, y, z = embed_torus(θ1, θ2, R, r)

    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Static torus surface (semi-transparent)
    u = np.linspace(0, 2*np.pi, torus_mesh_res)
    v = np.linspace(0, 2*np.pi, torus_mesh_res)
    U, V = np.meshgrid(u, v)
    X = (R + r*np.cos(U)) * np.cos(V)
    Y = (R + r*np.cos(U)) * np.sin(V)
    Z =  r * np.sin(U)
    ax.plot_surface(X, Y, Z, color='blue', alpha=surface_alpha,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)

    # Full trajectory (thin line for context)
    ax.plot(x, y, z, linewidth=0.8, alpha=0.4)

    # Moving particle
    particle, = ax.plot([], [], [], marker='o', markersize=6, linestyle='', color='orange')

    # Equal aspect
    L = R + r
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_title(rf"Resonant orbit {p}/{q} on $T^2$", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def init():
        particle.set_data([], [])
        particle.set_3d_properties([])
        return particle,

    def update(frame):
        particle.set_data([x[frame]], [y[frame]])
        particle.set_3d_properties([z[frame]])
        return particle,

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

    # Save animation as MP4
    out_path = Path(".") / fname
    writer = FFMpegWriter(fps=30, bitrate=1800)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path

# Create the MP4 animation
mp4_path = create_resonant_torus_animation(p=2, q=5, n_frames=600, fname="resonant_torus_2_5.mp4")
mp4_path
