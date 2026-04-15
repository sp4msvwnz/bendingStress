"""
Unsymmetric Bending Stress + 3D Deformation Visualizer
======================================================

Visualizes:
1. Stress distribution on a rectangular beam cross-section
2. Moment vector components
3. 3D deformed beam shape under unsymmetric bending

Requirements:
    pip install numpy matplotlib

Run:
    python unsymmetric_bending_3d.py

Suggested GitHub repo structure:
    unsymmetric-bending-visualizer/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── unsymmetric_bending_3d.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------------------------------------------------------
# Material / beam defaults
# -----------------------------------------------------------------------------
E = 200e9          # Young's modulus, Pa (steel-like default)
M = 300.0          # Applied moment, N·m
b_mm = 60.0        # Width, mm
h_mm = 90.0        # Height, mm
theta0 = 35.0      # Initial moment angle from z-axis, degrees
L0_mm = 900.0      # Beam length for 3D visualization, mm
DEF_SCALE0 = 250.0 # Visual amplification factor for 3D deformation


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
plt.style.use("dark_background")

fig = plt.figure(figsize=(16, 9), facecolor="#1e1e2e")
fig.canvas.manager.set_window_title("Unsymmetric Bending – 2D Stress + 3D Deformation")

ax_cs = fig.add_axes([0.05, 0.26, 0.33, 0.62])                 # 2D cross-section
ax_mom = fig.add_axes([0.41, 0.50, 0.16, 0.30])                # moment vector
ax_bar = fig.add_axes([0.36, 0.26, 0.018, 0.62])               # colorbar
ax_3d = fig.add_axes([0.60, 0.20, 0.36, 0.68], projection="3d")# 3D beam

for ax in [ax_cs, ax_mom, ax_bar]:
    ax.set_facecolor("#1e1e2e")
ax_3d.set_facecolor("#1e1e2e")


# -----------------------------------------------------------------------------
# Sliders
# -----------------------------------------------------------------------------
ax_angle = fig.add_axes([0.08, 0.15, 0.28, 0.025], facecolor="#2a2a3e")
ax_width = fig.add_axes([0.08, 0.11, 0.28, 0.025], facecolor="#2a2a3e")
ax_heigh = fig.add_axes([0.08, 0.07, 0.28, 0.025], facecolor="#2a2a3e")
ax_len   = fig.add_axes([0.08, 0.03, 0.28, 0.025], facecolor="#2a2a3e")
ax_scale = fig.add_axes([0.60, 0.08, 0.28, 0.025], facecolor="#2a2a3e")

sl_angle = Slider(ax_angle, "θ (°)", 0, 180, valinit=theta0, color="#EF9F27")
sl_width = Slider(ax_width, "Width b (mm)", 20, 120, valinit=b_mm, color="#5DCAA5")
sl_heigh = Slider(ax_heigh, "Height h (mm)", 20, 150, valinit=h_mm, color="#AFA9EC")
sl_len   = Slider(ax_len,   "Beam length L (mm)", 200, 2000, valinit=L0_mm, color="#6EC6FF")
sl_scale = Slider(ax_scale, "Deformation scale", 1, 1000, valinit=DEF_SCALE0, color="#FF7A90")

for sl in [sl_angle, sl_width, sl_heigh, sl_len, sl_scale]:
    sl.label.set_color("white")
    sl.valtext.set_color("white")


# -----------------------------------------------------------------------------
# Mechanics helpers
# -----------------------------------------------------------------------------
def section_properties_rectangular(b: float, h: float) -> tuple[float, float]:
    """
    Parameters
    ----------
    b : float
        Width in meters
    h : float
        Height in meters

    Returns
    -------
    Iy, Iz : tuple[float, float]
        Second moments of area in m^4
    """
    Iz = b * h**3 / 12.0
    Iy = h * b**3 / 12.0
    return Iy, Iz


def bending_stress(M: float, theta_deg: float, b: float, h: float, n: int = 300):
    """
    Returns grid coordinates and stress field over the cross-section.
    """
    theta = np.radians(theta_deg)
    Mz = M * np.cos(theta)
    My = M * np.sin(theta)

    Iy, Iz = section_properties_rectangular(b, h)

    z = np.linspace(-b / 2, b / 2, n)
    y = np.linspace(-h / 2, h / 2, n)
    Z, Y = np.meshgrid(z, y)

    sigma = (Mz / Iz) * Y - (My / Iy) * Z
    return Z, Y, sigma, My, Mz, Iy, Iz


def centerline_deflection(M: float, theta_deg: float, L: float, E: float, Iy: float, Iz: float, n: int = 80):
    """
    Small-deflection visualization for a cantilever-like beam under constant bending moment.
    This is used only for shape visualization.

    Curvatures:
        kappa_z = Mz / (E Iz)  -> bending in y
        kappa_y = My / (E Iy)  -> bending in z

    For constant curvature, deflection is quadratic in x:
        y(x) ~ 0.5 * kappa_z * x^2
        z(x) ~ -0.5 * kappa_y * x^2
    """
    theta = np.radians(theta_deg)
    Mz = M * np.cos(theta)
    My = M * np.sin(theta)

    x = np.linspace(0.0, L, n)

    kappa_z = Mz / (E * Iz)
    kappa_y = My / (E * Iy)

    y_def = 0.5 * kappa_z * x**2
    z_def = -0.5 * kappa_y * x**2

    return x, y_def, z_def, kappa_y, kappa_z


def build_deformed_beam_surface(x, y_c, z_c, b, h, scale=1.0):
    """
    Build a simple deformed rectangular beam surface by extruding the cross-section
    along the deformed centerline.
    """
    yy = np.array([-h / 2, h / 2])
    zz = np.array([-b / 2, b / 2])

    X = np.zeros((2, len(x), 2))
    Y = np.zeros((2, len(x), 2))
    Z = np.zeros((2, len(x), 2))

    for i, xi in enumerate(x):
        for j, y_local in enumerate(yy):
            for k, z_local in enumerate(zz):
                X[j, i, k] = xi
                Y[j, i, k] = scale * y_c[i] + y_local
                Z[j, i, k] = scale * z_c[i] + z_local

    return X, Y, Z


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw(theta_deg: float, b_mm_val: float, h_mm_val: float, L_mm_val: float, def_scale: float):
    ax_cs.cla()
    ax_mom.cla()
    ax_bar.cla()
    ax_3d.cla()

    for ax in [ax_cs, ax_mom]:
        ax.set_facecolor("#1e1e2e")
    ax_3d.set_facecolor("#1e1e2e")

    # Convert to SI
    b = b_mm_val * 1e-3
    h = h_mm_val * 1e-3
    L = L_mm_val * 1e-3

    # Stress field
    Zg, Yg, sigma, My, Mz, Iy, Iz = bending_stress(M, theta_deg, b, h, n=300)
    sigma_MPa = sigma / 1e6

    max_abs = np.max(np.abs(sigma_MPa))
    if max_abs < 1e-12:
        max_abs = 1.0

    # -------------------------------------------------------------------------
    # 2D cross-section stress plot
    # -------------------------------------------------------------------------
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cm = ax_cs.pcolormesh(
        Zg * 1e3,
        Yg * 1e3,
        sigma_MPa,
        cmap="RdBu_r",
        norm=norm,
        shading="auto"
    )

    rect = mpatches.FancyBboxPatch(
        (-b / 2 * 1e3, -h / 2 * 1e3),
        b * 1e3,
        h * 1e3,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor="white",
        facecolor="none"
    )
    ax_cs.add_patch(rect)

    # Neutral axis
    na_slope = (My * Iz) / (Mz * Iy) if abs(Mz) > 1e-12 else np.inf
    z_range = np.array([-b / 2 * 1e3, b / 2 * 1e3])

    if np.isinf(na_slope):
        ax_cs.axvline(0, color="gray", lw=1.6, ls="--", label="Neutral axis", alpha=0.9)
    else:
        y_na = na_slope * z_range
        ax_cs.plot(
            z_range,
            np.clip(y_na, -h / 2 * 1e3, h / 2 * 1e3),
            color="gray",
            lw=1.8,
            ls="--",
            label="Neutral axis",
            alpha=0.9
        )

    ax_cs.axhline(0, color="white", lw=0.5, ls=":", alpha=0.35)
    ax_cs.axvline(0, color="white", lw=0.5, ls=":", alpha=0.35)

    corners = {
        "A": (-b / 2 * 1e3,  h / 2 * 1e3),
        "B": ( b / 2 * 1e3,  h / 2 * 1e3),
        "C": ( b / 2 * 1e3, -h / 2 * 1e3),
        "D": (-b / 2 * 1e3, -h / 2 * 1e3),
    }
    offsets = {
        "A": (-6,  6, "right", "bottom"),
        "B": ( 6,  6, "left",  "bottom"),
        "C": ( 6, -6, "left",  "top"),
        "D": (-6, -6, "right", "top"),
    }

    for name, (zc, yc) in corners.items():
        s = (Mz / Iz) * (yc * 1e-3) - (My / Iy) * (zc * 1e-3)
        s_mpa = s / 1e6
        col = "#E24B4A" if s_mpa > 0 else ("#378ADD" if s_mpa < 0 else "gray")
        dz, dy, ha, va = offsets[name]
        sign = "+" if s_mpa >= 0 else ""
        ax_cs.plot(zc, yc, "o", color=col, ms=7, zorder=5)
        ax_cs.text(
            zc + dz,
            yc + dy,
            f"{name}\n{sign}{s_mpa:.2f} MPa",
            color=col,
            fontsize=8.5,
            ha=ha,
            va=va,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#1e1e2e", alpha=0.6, ec="none")
        )

    ax_cs.set_xlim(-b / 2 * 1e3 - 20, b / 2 * 1e3 + 20)
    ax_cs.set_ylim(-h / 2 * 1e3 - 20, h / 2 * 1e3 + 20)
    ax_cs.set_aspect("equal")
    ax_cs.set_xlabel("z (mm)", color="white")
    ax_cs.set_ylabel("y (mm)", color="white")
    ax_cs.tick_params(colors="white")
    for sp in ax_cs.spines.values():
        sp.set_color("#444")

    ax_cs.set_title(
        f"Cross-section stress | θ = {theta_deg:.1f}° | σ_max = {max_abs:.2f} MPa",
        color="white",
        fontsize=11,
        pad=10
    )
    ax_cs.legend(
        loc="upper center",
        fontsize=8,
        facecolor="#2a2a3e",
        edgecolor="#555",
        labelcolor="white"
    )

    cb = plt.colorbar(cm, cax=ax_bar)
    cb.set_label("σ (MPa)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # -------------------------------------------------------------------------
    # Moment vector diagram
    # -------------------------------------------------------------------------
    ax_mom.set_aspect("equal")
    ax_mom.set_xlim(-1.4, 1.4)
    ax_mom.set_ylim(-1.4, 1.4)
    ax_mom.tick_params(colors="white")
    for sp in ax_mom.spines.values():
        sp.set_color("#444")

    myn = My / M if M != 0 else 0.0
    mzn = Mz / M if M != 0 else 0.0

    kw = dict(head_width=0.07, head_length=0.07, length_includes_head=True)

    if abs(myn) > 0.03:
        ax_mom.arrow(0, 0, myn, 0, color="#5DCAA5", **kw, alpha=0.85)
        ax_mom.text(myn + 0.06, 0, f"My={My:.1f} N·m", color="#5DCAA5", fontsize=7.5, va="center")

    if abs(mzn) > 0.03:
        ax_mom.arrow(0, 0, 0, mzn, color="#AFA9EC", **kw, alpha=0.85)
        ax_mom.text(0.06, mzn + 0.06, f"Mz={Mz:.1f} N·m", color="#AFA9EC", fontsize=7.5, ha="left")

    ax_mom.arrow(0, 0, myn, mzn, color="#EF9F27", **kw, lw=1.6)
    ax_mom.text(myn / 2, mzn / 2 + 0.12, "M", color="#EF9F27", fontsize=10, fontweight="bold", ha="center")

    ax_mom.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax_mom.axvline(0, color="white", lw=0.5, alpha=0.3)
    ax_mom.text(1.28, 0, "z →", color="white", fontsize=8, va="center")
    ax_mom.text(0, 1.28, "y ↑", color="white", fontsize=8, ha="center")

    theta_rad = np.radians(theta_deg)
    arc_angles = np.linspace(0, theta_rad, 60)
    ax_mom.plot(0.5 * np.cos(arc_angles), 0.5 * np.sin(arc_angles), color="#EF9F27", lw=1, alpha=0.7)
    mid = theta_rad / 2
    ax_mom.text(0.62 * np.cos(mid), 0.62 * np.sin(mid), f"θ={theta_deg:.1f}°", color="#EF9F27", fontsize=7.5, ha="center")
    ax_mom.set_title("Moment components", color="white", fontsize=9, pad=6)

    # -------------------------------------------------------------------------
    # 3D deformation
    # -------------------------------------------------------------------------
    x, y_def, z_def, kappa_y, kappa_z = centerline_deflection(M, theta_deg, L, E, Iy, Iz, n=80)

    # Centerline
    ax_3d.plot(
        x * 1e3,
        (def_scale * y_def) * 1e3,
        (def_scale * z_def) * 1e3,
        lw=2.6,
        color="#EF9F27",
        label="Deformed centerline"
    )

    # Undeformed reference
    ax_3d.plot(
        x * 1e3,
        np.zeros_like(x),
        np.zeros_like(x),
        lw=1.0,
        ls="--",
        color="white",
        alpha=0.5,
        label="Undeformed centerline"
    )

    # Surface extrusion
    Xsurf, Ysurf, Zsurf = build_deformed_beam_surface(x, y_def, z_def, b, h, scale=def_scale)

    # top / bottom faces
    ax_3d.plot_surface(Xsurf[1] * 1e3, Ysurf[1] * 1e3, Zsurf[1] * 1e3, alpha=0.25, linewidth=0)
    ax_3d.plot_surface(Xsurf[0] * 1e3, Ysurf[0] * 1e3, Zsurf[0] * 1e3, alpha=0.25, linewidth=0)

    # side faces
    ax_3d.plot_surface(Xsurf[:, :, 0] * 1e3, Ysurf[:, :, 0] * 1e3, Zsurf[:, :, 0] * 1e3, alpha=0.18, linewidth=0)
    ax_3d.plot_surface(Xsurf[:, :, 1] * 1e3, Ysurf[:, :, 1] * 1e3, Zsurf[:, :, 1] * 1e3, alpha=0.18, linewidth=0)

    ax_3d.set_title("3D deformation view", color="white", pad=12)
    ax_3d.set_xlabel("x (mm)", color="white", labelpad=8)
    ax_3d.set_ylabel("y deflection (mm)", color="white", labelpad=8)
    ax_3d.set_zlabel("z deflection (mm)", color="white", labelpad=8)
    ax_3d.tick_params(colors="white")

    # Improve aspect
    x_span = max(L_mm_val, 1.0)
    y_span = max(np.ptp((def_scale * y_def) * 1e3) + h_mm_val, h_mm_val)
    z_span = max(np.ptp((def_scale * z_def) * 1e3) + b_mm_val, b_mm_val)
    max_span = max(x_span, y_span, z_span)

    ax_3d.set_xlim(0, x_span)
    ax_3d.set_ylim(-max_span * 0.15, max_span * 0.15)
    ax_3d.set_zlim(-max_span * 0.15, max_span * 0.15)
    ax_3d.view_init(elev=22, azim=-58)

    # Make panes subtle
    try:
        ax_3d.xaxis.pane.set_alpha(0.08)
        ax_3d.yaxis.pane.set_alpha(0.08)
        ax_3d.zaxis.pane.set_alpha(0.08)
    except Exception:
        pass

    ax_3d.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------------------
    # Info box
    # -------------------------------------------------------------------------
    na_ang = np.degrees(np.arctan(na_slope)) if not np.isinf(na_slope) else 90.0
    tip_y_mm = (def_scale * y_def[-1]) * 1e3
    tip_z_mm = (def_scale * z_def[-1]) * 1e3

    info_txt = (
        f"Iz = {Iz:.4e} m^4\n"
        f"Iy = {Iy:.4e} m^4\n"
        f"NA angle from z-axis = {na_ang:.2f}°\n"
        f"kappa_z = {kappa_z:.4e} 1/m\n"
        f"kappa_y = {kappa_y:.4e} 1/m\n"
        f"Tip y-defl (scaled) = {tip_y_mm:.2f} mm\n"
        f"Tip z-defl (scaled) = {tip_z_mm:.2f} mm"
    )

    fig.texts = [t for t in fig.texts if not getattr(t, "_info", False)]
    txt = fig.text(
        0.41,
        0.30,
        info_txt,
        color="white",
        fontsize=8.5,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#2a2a3e", ec="#555")
    )
    txt._info = True

    fig.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------
def on_change(_):
    draw(
        sl_angle.val,
        sl_width.val,
        sl_heigh.val,
        sl_len.val,
        sl_scale.val
    )


sl_angle.on_changed(on_change)
sl_width.on_changed(on_change)
sl_heigh.on_changed(on_change)
sl_len.on_changed(on_change)
sl_scale.on_changed(on_change)


# -----------------------------------------------------------------------------
# Initial render
# -----------------------------------------------------------------------------
draw(theta0, b_mm, h_mm, L0_mm, DEF_SCALE0)

plt.suptitle(
    "Unsymmetric Bending – Stress Distribution and 3D Deformation",
    color="white",
    fontsize=14,
    y=0.97
)

plt.show()