import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend (remove if you
                                          # want a live preview in Jupyter)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  –– enables 3-D projection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def motion_angles(p, q, n):
    """Return θ₁(t), θ₂(t) for n evenly spaced samples in [0,2π]."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return (p * t) % (2 * np.pi), (q * t) % (2 * np.pi)


def to_xyz(theta1, theta2, R=3.0, r=1.0):
    """Embed angle pair → (x,y,z) on the torus."""
    x = (R + r * np.cos(theta1)) * np.cos(theta2)
    y = (R + r * np.cos(theta1)) * np.sin(theta2)
    z = r * np.sin(theta1)
    return x, y, z


# ------------------------------------------------------------
# Animation creator
# ------------------------------------------------------------
def make_anim_mp4(
    p=2,
    q=5,
    frames=300,
    R=3.0,
    r=1.0,
    mesh=45,
    alpha=0.25,
    fname="torus_trace_2_5.mp4",
):
    """Create an MP4 of a resonant orbit on a torus *with a live trace*."""
    # Orbit samples
    th1, th2 = motion_angles(p, q, frames)
    x, y, z = to_xyz(th1, th2, R=R, r=r)

    # ── Figure & torus surface ──────────────────────────────
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, mesh)
    v = np.linspace(0, 2 * np.pi, mesh)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(U)) * np.cos(V)
    Y = (R + r * np.cos(U)) * np.sin(V)
    Z = r * np.sin(U)
    ax.plot_surface(X, Y, Z, color="blue", alpha=alpha, linewidth=0)

    # ── Static full orbit (faint) for context ───────────────
    ax.plot(x, y, z, lw=0.6, alpha=0.2, color="orange")

    # ── Dynamic artists: particle + trace ───────────────────
    particle, = ax.plot([], [], [], "o", color="orange", markersize=5)
    trace,    = ax.plot([], [], [], lw=1.4, color="orange")  # grows each frame

    # ── Axis cosmetics ──────────────────────────────────────
    L = R + r
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.axis("off")

    # ── Animation callbacks ─────────────────────────────────
    def init():
        particle.set_data([], [])
        particle.set_3d_properties([])
        trace.set_data([], [])
        trace.set_3d_properties([])
        return particle, trace

    def update(i):
        # Move particle
        particle.set_data([x[i]], [y[i]])
        particle.set_3d_properties([z[i]])
        # Extend trace up to frame i
        trace.set_data(x[: i + 1], y[: i + 1])
        trace.set_3d_properties(z[: i + 1])
        return particle, trace

    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=20
    )

    # ── Save to MP4 ─────────────────────────────────────────
    out_path = Path(".") / fname
    ani.save(out_path, writer=FFMpegWriter(fps=30, bitrate=1500))
    plt.close(fig)
    return out_path


# Example (don’t run automatically in this snippet):
mp4_path = make_anim_mp4()
