import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

# ------------------------------------------------------------------
# 1.  A Weierstrass–type fractal curve on [0,1]
# ------------------------------------------------------------------
def weierstrass(t, a=3.0, b=0.5, n_terms=40):
    """
    Return W(t) = sum_{k=0}^{n_terms-1} b**k * sin(a**k * pi * t).
    For 1 < a and 0 < b < 1, the graph has Hausdorff dimension
        D = 2 + log(b)/log(a)  in (1, 2).
    """
    t = np.asarray(t)
    w = np.zeros_like(t)
    coeffs = b ** np.arange(n_terms)
    freqs  = a ** np.arange(n_terms)
    for c, f in zip(coeffs, freqs):
        w += c * np.sin(np.pi * f * t)
    return w


# ------------------------------------------------------------------
# 2.  Fractal trajectory on the 2-torus
# ------------------------------------------------------------------
def fractal_torus_motion(n_points=20_000, a=3.0, b=0.5, n_terms=40,
                         phi=(np.sqrt(5)-1)/2):
    t = np.linspace(0.0, 1.0, n_points, endpoint=False)

    theta1 = 2*np.pi * t
    # pass n_terms through
    theta2 = 2*np.pi * (phi * t + weierstrass(t, a=a, b=b, n_terms=n_terms))
    theta1 = np.mod(theta1, 2*np.pi)
    theta2 = np.mod(theta2, 2*np.pi)
    return theta1, theta2


# ------------------------------------------------------------------
# 3.  Embed on a standard torus in ℝ³
# ------------------------------------------------------------------
def embed_torus(theta1, theta2, R=3.0, r=1.0):
    """
    (θ1, θ2) ↦ (x, y, z) with major radius R and minor radius r.
    """
    x = (R + r * np.cos(theta1)) * np.cos(theta2)
    y = (R + r * np.cos(theta1)) * np.sin(theta2)
    z =  r * np.sin(theta1)
    return x, y, z


# ------------------------------------------------------------------
# 4.  Put everything together and draw
# ------------------------------------------------------------------
def plot_fractal_torus(n_points=20_000, a=3.0, b=0.5, n_terms=40,
                       R=3.0, r=1.0, mesh_res=60, alpha=0.25):
    """
    Visualise a fractal-dimension trajectory on the torus.
    Dimension D = 2 + log(b)/log(a)  ∈ (1, 2).
    """
    θ1, θ2 = fractal_torus_motion(n_points=n_points, a=a, b=b,
                                  n_terms=n_terms)
    x, y, z = embed_torus(θ1, θ2, R, r)

    # ---- basic figure ----
    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # ---- semi-transparent torus surface ----
    u = np.linspace(0, 2*np.pi, mesh_res)
    v = np.linspace(0, 2*np.pi, mesh_res)
    U, V = np.meshgrid(u, v)
    X = (R + r*np.cos(U)) * np.cos(V)
    Y = (R + r*np.cos(U)) * np.sin(V)
    Z =  r * np.sin(U)
    ax.plot_surface(X, Y, Z, alpha=alpha,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)

    # ---- overlay the fractal path ----
    ax.plot(x, y, z, linewidth=1.2)

    # ---- cosmetics ----
    ax.set_title(rf"Fractal orbit on $T^2$ ($D\!\approx\!{2+np.log(b)/np.log(a):.2f}$)",
                 pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    L = R + r
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Example: dimension ≈ 1.37 (a = 3, b = 0.5)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # plot_fractal_torus(a=3.0, b=0.5)

    # dimension ≈ 1.85, needs more samples
    plot_fractal_torus(a=2.0, b=0.75, n_terms=50, n_points=35_000)

    # crisper, lower-dim fractal
    plot_fractal_torus(a=4.0, b=0.30, n_terms=35, n_points=18_000)

