from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Unsymmetric Bending Visualizer",
    layout="wide",
)

st.title("Unsymmetric Bending - Interactive 3D Visualizer")
st.caption("Browser-based stress and deformation viewer for a rectangular beam.")


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    E_GPa = st.slider("Young's modulus E (GPa)", 1.0, 300.0, 200.0, 1.0)
    M = st.slider("Applied moment M (N·m)", 0.0, 2000.0, 300.0, 10.0)
    theta_deg = st.slider("Moment angle θ (deg)", 0.0, 180.0, 35.0, 1.0)

    b_mm = st.slider("Width b (mm)", 20.0, 200.0, 60.0, 1.0)
    h_mm = st.slider("Height h (mm)", 20.0, 200.0, 90.0, 1.0)
    L_mm = st.slider("Beam length L (mm)", 200.0, 3000.0, 900.0, 10.0)

    def_scale = st.slider("Deformation display scale", 1.0, 3000.0, 250.0, 1.0)
    n_x = st.slider("3D resolution", 20, 120, 60, 5)
    n_cs = st.slider("Cross-section grid", 40, 250, 120, 10)

    show_surface = st.checkbox("Show 3D beam surface", value=True)
    show_centerline = st.checkbox("Show deformed centerline", value=True)
    show_undeformed = st.checkbox("Show undeformed centerline", value=True)


# -----------------------------------------------------------------------------
# Mechanics
# -----------------------------------------------------------------------------
E = E_GPa * 1e9
b = b_mm * 1e-3
h = h_mm * 1e-3
L = L_mm * 1e-3
theta = np.radians(theta_deg)

Mz = M * np.cos(theta)
My = M * np.sin(theta)

Iz = b * h**3 / 12.0
Iy = h * b**3 / 12.0

# Cross-section grid
z = np.linspace(-b / 2, b / 2, n_cs)
y = np.linspace(-h / 2, h / 2, n_cs)
Z, Y = np.meshgrid(z, y)

# Unsymmetric bending stress
sigma = (Mz / Iz) * Y - (My / Iy) * Z
sigma_mpa = sigma / 1e6
sigma_max = float(np.max(np.abs(sigma_mpa))) if np.max(np.abs(sigma_mpa)) > 1e-12 else 1.0

# Neutral axis: (Mz/Iz) y - (My/Iy) z = 0  => y = (My*Iz)/(Mz*Iy) z
na_slope = np.inf if abs(Mz) < 1e-12 else (My * Iz) / (Mz * Iy)
na_angle_deg = 90.0 if np.isinf(na_slope) else np.degrees(np.arctan(na_slope))

# Beam centerline, simple constant-curvature visualization
x = np.linspace(0.0, L, n_x)
kappa_z = 0.0 if Iz == 0 else Mz / (E * Iz)   # bending causing y deflection
kappa_y = 0.0 if Iy == 0 else My / (E * Iy)   # bending causing z deflection

y_def = 0.5 * kappa_z * x**2
z_def = -0.5 * kappa_y * x**2

y_vis = def_scale * y_def
z_vis = def_scale * z_def

tip_y_mm = y_vis[-1] * 1e3
tip_z_mm = z_vis[-1] * 1e3


# -----------------------------------------------------------------------------
# Helpers for 3D beam
# -----------------------------------------------------------------------------
def add_edge(fig: go.Figure, xvals, yvals, zvals, name: str, width: int = 4, showlegend: bool = False):
    fig.add_trace(
        go.Scatter3d(
            x=xvals,
            y=yvals,
            z=zvals,
            mode="lines",
            name=name,
            showlegend=showlegend,
            line=dict(width=width),
        )
    )


def beam_corner_lines(x_arr, y_center, z_center, b_val, h_val):
    y_offsets = np.array([-h_val / 2, h_val / 2])
    z_offsets = np.array([-b_val / 2, b_val / 2])

    lines = []
    for yo in y_offsets:
        for zo in z_offsets:
            lines.append((
                x_arr * 1e3,
                (y_center + yo) * 1e3,
                (z_center + zo) * 1e3,
            ))
    return lines


def make_surface(x_arr, y_center, z_center, b_val, h_val):
    yy = np.array([-h_val / 2, h_val / 2])
    zz = np.array([-b_val / 2, b_val / 2])

    X = np.zeros((2, len(x_arr), 2))
    Ysurf = np.zeros((2, len(x_arr), 2))
    Zsurf = np.zeros((2, len(x_arr), 2))

    for i, xi in enumerate(x_arr):
        for j, yo in enumerate(yy):
            for k, zo in enumerate(zz):
                X[j, i, k] = xi
                Ysurf[j, i, k] = y_center[i] + yo
                Zsurf[j, i, k] = z_center[i] + zo

    return X * 1e3, Ysurf * 1e3, Zsurf * 1e3


# -----------------------------------------------------------------------------
# 2D stress plot
# -----------------------------------------------------------------------------
heatmap = go.Figure()

heatmap.add_trace(
    go.Heatmap(
        x=z * 1e3,
        y=y * 1e3,
        z=sigma_mpa,
        colorscale="RdBu",
        zmid=0,
        zmin=-sigma_max,
        zmax=sigma_max,
        colorbar=dict(title="σ (MPa)"),
        hovertemplate="z = %{x:.2f} mm<br>y = %{y:.2f} mm<br>σ = %{z:.3f} MPa<extra></extra>",
    )
)

# Rectangle outline
rect_z = np.array([-b / 2, b / 2, b / 2, -b / 2, -b / 2]) * 1e3
rect_y = np.array([-h / 2, -h / 2, h / 2, h / 2, -h / 2]) * 1e3
heatmap.add_trace(
    go.Scatter(
        x=rect_z,
        y=rect_y,
        mode="lines",
        name="Section boundary",
        line=dict(color="white", width=2),
    )
)

# Neutral axis
z_line = np.linspace(-b / 2, b / 2, 200) * 1e3
if np.isinf(na_slope):
    heatmap.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-h / 2 * 1e3, h / 2 * 1e3],
            mode="lines",
            name="Neutral axis",
            line=dict(color="gray", dash="dash"),
        )
    )
else:
    y_line = na_slope * z_line
    heatmap.add_trace(
        go.Scatter(
            x=z_line,
            y=np.clip(y_line, -h / 2 * 1e3, h / 2 * 1e3),
            mode="lines",
            name="Neutral axis",
            line=dict(color="gray", dash="dash"),
        )
    )

# Corner labels
corners = {
    "A": (-b / 2,  h / 2),
    "B": ( b / 2,  h / 2),
    "C": ( b / 2, -h / 2),
    "D": (-b / 2, -h / 2),
}
for name, (zc, yc) in corners.items():
    s = (Mz / Iz) * yc - (My / Iy) * zc
    heatmap.add_trace(
        go.Scatter(
            x=[zc * 1e3],
            y=[yc * 1e3],
            mode="markers+text",
            text=[f"{name}<br>{s/1e6:+.2f} MPa"],
            textposition="top center",
            name=name,
            marker=dict(size=8),
            showlegend=False,
        )
    )

heatmap.update_layout(
    title=f"Cross-section Stress Distribution (θ = {theta_deg:.1f}°)",
    xaxis_title="z (mm)",
    yaxis_title="y (mm)",
    template="plotly_dark",
    height=600,
    yaxis=dict(scaleanchor="x", scaleratio=1),
)


# -----------------------------------------------------------------------------
# Moment vector plot
# -----------------------------------------------------------------------------
moment_fig = go.Figure()

moment_fig.add_trace(
    go.Scatter(
        x=[0, My],
        y=[0, Mz],
        mode="lines+markers+text",
        text=["", "M"],
        textposition="top center",
        name="Resultant moment",
    )
)

moment_fig.add_trace(
    go.Scatter(
        x=[0, My],
        y=[0, 0],
        mode="lines+markers",
        name="My component",
    )
)

moment_fig.add_trace(
    go.Scatter(
        x=[0, 0],
        y=[0, Mz],
        mode="lines+markers",
        name="Mz component",
    )
)

moment_fig.update_layout(
    title="Moment Components",
    xaxis_title="My (N·m)",
    yaxis_title="Mz (N·m)",
    template="plotly_dark",
    height=320,
)


# -----------------------------------------------------------------------------
# 3D deformation plot
# -----------------------------------------------------------------------------
beam_fig = go.Figure()

if show_undeformed:
    beam_fig.add_trace(
        go.Scatter3d(
            x=x * 1e3,
            y=np.zeros_like(x),
            z=np.zeros_like(x),
            mode="lines",
            name="Undeformed centerline",
            line=dict(width=4, dash="dash"),
        )
    )

if show_centerline:
    beam_fig.add_trace(
        go.Scatter3d(
            x=x * 1e3,
            y=y_vis * 1e3,
            z=z_vis * 1e3,
            mode="lines",
            name="Deformed centerline",
            line=dict(width=7),
        )
    )

for i, line_data in enumerate(beam_corner_lines(x, y_vis, z_vis, b, h)):
    add_edge(
        beam_fig,
        line_data[0],
        line_data[1],
        line_data[2],
        name=f"Edge {i+1}",
        width=3,
        showlegend=False,
    )

if show_surface:
    Xs, Ys, Zs = make_surface(x, y_vis, z_vis, b, h)

    beam_fig.add_trace(
        go.Surface(
            x=Xs[0],
            y=Ys[0],
            z=Zs[0],
            showscale=False,
            opacity=0.35,
            name="Bottom/Top face",
        )
    )
    beam_fig.add_trace(
        go.Surface(
            x=Xs[1],
            y=Ys[1],
            z=Zs[1],
            showscale=False,
            opacity=0.35,
            name="Bottom/Top face 2",
        )
    )
    beam_fig.add_trace(
        go.Surface(
            x=Xs[:, :, 0],
            y=Ys[:, :, 0],
            z=Zs[:, :, 0],
            showscale=False,
            opacity=0.25,
            name="Side face",
        )
    )
    beam_fig.add_trace(
        go.Surface(
            x=Xs[:, :, 1],
            y=Ys[:, :, 1],
            z=Zs[:, :, 1],
            showscale=False,
            opacity=0.25,
            name="Side face 2",
        )
    )

beam_fig.update_layout(
    title="3D Deformation View",
    template="plotly_dark",
    height=700,
    scene=dict(
        xaxis_title="x (mm)",
        yaxis_title="y deflection (mm)",
        zaxis_title="z deflection (mm)",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Iy (m^4)", f"{Iy:.3e}")
c2.metric("Iz (m^4)", f"{Iz:.3e}")
c3.metric("NA angle", f"{na_angle_deg:.2f} deg")
c4.metric("Tip y deflection", f"{tip_y_mm:.2f} mm")
c5.metric("Tip z deflection", f"{tip_z_mm:.2f} mm")

st.markdown(
    """
This 3D shape is a **visualization of bending deformation** using small-deflection beam curvature.
The deformation is amplified by the display scale so you can inspect direction and relative behavior.
"""
)

col1, col2 = st.columns([1.1, 1.3])

with col1:
    st.plotly_chart(heatmap, use_container_width=True)
    st.plotly_chart(moment_fig, use_container_width=True)

with col2:
    st.plotly_chart(beam_fig, use_container_width=True)