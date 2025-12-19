# pytrajviz/app.py
# Interactive web-based molecular dynamics trajectory viewer using Plotly Dash
# Features: 3D real-time visualization, RMSD plot, frame navigation, play/pause, GIF export
# Author: Jakub Hryc (2025–2026)
# GitHub: https://github.com/Charnel2500/pytrajviz

import mdtraj as md
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from argparse import ArgumentParser
import imageio.v3 as iio
import os


# CPK color scheme – standard in molecular visualization
CPK_COLORS = {
    'H': '#FFFFFF',  # white
    'C': '#909090',  # light gray
    'O': '#FF0D0D',  # red
    'N': '#3050F8',  # blue
    'P': '#FFA500',  # orange
    'S': '#FFFF30',  # yellow
    'CL': '#1FF01F', 'Cl': '#1FF01F',  # green
    'NA': '#AB5CF2', 'Na': '#AB5CF2',  # purple
    'MG': '#8AFF00', 'Mg': '#8AFF00',  # bright green
}


def get_atom_color(atom):
    """Return CPK color for a given atom element."""
    element = atom.element.symbol.upper()
    return CPK_COLORS.get(element, '#CCCCCC')  # fallback: light gray


def compute_rmsd(traj):
    """
    Compute RMSD relative to the first frame.
    Tries: C-alpha → backbone → all atoms (fallback).
    """
    ref = traj[0]
    for selection in ["name CA", "backbone", "all"]:
        indices = traj.topology.select(selection)
        if len(indices) > 0:
            return md.rmsd(traj, ref, atom_indices=indices)
    return np.zeros(len(traj))  # fallback if nothing matches


class TrajVizApp:
    """
    Main application class for interactive MD trajectory visualization.
    Loads trajectory, prepares data, builds and runs Dash web app.
    """
    def __init__(self, traj_file, top_file):
        # Load trajectory using MDTraj
        self.traj = md.load(traj_file, top=top_file)
        self.rmsd = compute_rmsd(self.traj)
        self.n_frames = len(self.traj)

        # Precompute atom colors and hover labels for performance
        self.colors = [get_atom_color(atom) for atom in self.traj.topology.atoms]
        self.atom_labels = [
            f"{atom.name} ({atom.residue.name}{atom.residue.resSeq})"
            for atom in self.traj.topology.atoms
        ]

        print(f"[pyTrajViz] Successfully loaded trajectory")
        print(f"    → Frames: {self.n_frames}")
        print(f"    → Atoms:  {self.traj.n_atoms}")
        print(f"    → RMSD range: {self.rmsd.min():.3f} – {self.rmsd.max():.3f} Å")

    def _make_frame_figure(self, frame_idx):
        """Generate 3D Scatter plot for a specific frame."""
        pos = self.traj.xyz[frame_idx] * 10  # convert nm → Å for better scaling

        fig = go.Figure(data=[go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker=dict(size=5, color=self.colors, opacity=0.9),
            text=self.atom_labels,
            hovertemplate=(
                '<b>%{text}</b><br>'
                'X: %{x:.2f} Å<br>'
                'Y: %{y:.2f} Å<br>'
                'Z: %{z:.2f} Å<br>'
                '<extra></extra>'
            )
        )])

        fig.update_layout(
            scene=dict(aspectmode='data', xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
            height=800,
            title=f"Frame {frame_idx} | RMSD: {self.rmsd[frame_idx]:.3f} Å",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    def build_app(self):
        """Build and configure the Dash web application."""
        app = dash.Dash(__name__, title="pyTrajViz – Interactive MD Viewer")

        # Initial figures
        initial_3d = self._make_frame_figure(0)
        rmsd_plot = px.line(
            x=np.arange(self.n_frames),
            y=self.rmsd,
            labels={"x": "Frame", "y": "RMSD (Å)"},
            title="RMSD over Time"
        ).update_layout(height=300)

        # Layout
        app.layout = html.Div([
            html.H1("pyTrajViz – Interactive MD Trajectory Explorer",
                    style={'textAlign': 'center', 'margin': '20px'}),

            # Control buttons
            html.Div([
                html.Button("Play", id="play-btn", n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px'}),
                html.Button("Pause", id="pause-btn", n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px', 'display': 'none'}),
                html.Button("Export GIF", id="gif-btn", n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px'}),
                dcc.Download(id="download-gif")
            ], style={'textAlign': 'center', 'margin': '20px'}),

            # RMSD plot
            dcc.Graph(id="rmsd-plot", figure=rmsd_plot),

            # Frame slider
            html.Div(f"Total frames: {self.n_frames} | RMSD range: {self.rmsd.min():.3f}–{self.rmsd.max():.3f} Å",
                     style={'textAlign': 'center', 'fontSize': 16, 'margin': '10px'}),
            dcc.Slider(
                id="frame-slider",
                min=0, max=self.n_frames-1, value=0, step=1,
                marks={i: str(i) for i in range(0, self.n_frames, max(1, self.n_frames//10))}
            ),

            # Auto-play interval
            dcc.Interval(id="interval", interval=100, n_intervals=0, disabled=True),

            # 3D viewer
            dcc.Graph(id="3d-view", figure=initial_3d),

            # Info bar
            html.Div(id="info", style={'textAlign': 'center', 'fontSize': 18, 'margin': '10px'})
        ])

        # Main callback – handles slider, play/pause, and frame updates
        @app.callback(
            [Output("3d-view", "figure"), Output("info", "children"),
             Output("interval", "disabled"), Output("play-btn", "style"), Output("pause-btn", "style")],
            [Input("frame-slider", "value"), Input("interval", "n_intervals"),
             Input("play-btn", "n_clicks"), Input("pause-btn", "n_clicks")],
            State("interval", "disabled")
        )
        def update_visualization(frame, n_intervals, play_clicks, pause_clicks, currently_playing):
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

            playing = currently_playing

            if triggered_id == "play-btn":
                playing = False
            elif triggered_id == "pause-btn":
                playing = True
            elif triggered_id == "interval" and not playing:
                frame = (frame + 1) % self.n_frames
                playing = False

            fig = self._make_frame_figure(frame)
            info = f"Frame: {frame} | RMSD: {self.rmsd[frame]:.3f} Å | Animation: {'ON' if not playing else 'OFF'}"

            play_style = {'display': 'none'} if not playing else {}
            pause_style = {'display': 'none'} if playing else {}

            return fig, info, playing, play_style, pause_style

        # GIF export callback
        @app.callback(
            Output("download-gif", "data"),
            Input("gif-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def export_to_gif(n_clicks):
            if not n_clicks:
                return dash.no_update

            print("[pyTrajViz] Exporting trajectory to GIF...")
            frames = []
            step = max(1, self.n_frames // 100)
            for i in range(0, self.n_frames, step):
                fig = self._make_frame_figure(i)
                img_bytes = fig.to_image(format="png")
                frames.append(iio.imread(img_bytes))

            filename = "pytrajviz_trajectory.gif"
            iio.imwrite(filename, frames, duration=100, loop=0)
            print(f"[pyTrajViz] GIF saved as {filename}")
            return dcc.send_file(filename)

        return app


def main():
    """Command-line entry point."""
    parser = ArgumentParser(description="pyTrajViz – Interactive MD trajectory web viewer")
    parser.add_argument("--traj", required=True, help="Trajectory file (.xtc, .dcd, .nc, ...)")
    parser.add_argument("--top", required=True, help="Topology file (.gro, .pdb, ...)")
    parser.add_argument("--port", type=int, default=8050, help="Web server port")
    args = parser.parse_args()

    viz = TrajVizApp(args.traj, args.top)
    app = viz.build_app()
    print(f"[pyTrajViz] Server starting → http://127.0.0.1:{args.port}")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
