# pytrajviz/app.py
# Interactive web visualization for MD trajectories using Plotly Dash
# Features: 3D trajectory viewer, RMSD plot, frame slider, hover info
# Load .xtc + .gro, run: pytrajviz run --traj traj.xtc --top top.gro

import mdtraj as md
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import numpy as np
from argparse import ArgumentParser
import sys

def compute_rmsd(traj):
    """Compute RMSD per frame (backbone)."""
    ref = traj[0]
    rmsd = md.rmsd(traj, ref, atom_indices=traj.topology.select("name CA"))
    return rmsd

class TrajVizApp:
    def __init__(self, traj_file, top_file):
        self.traj = md.load(traj_file, top=top_file)
        self.rmsd = compute_rmsd(self.traj)
        self.n_frames = len(self.traj)
        print(f"[pyTrajViz] Loaded {self.n_frames} frames, {self.traj.n_atoms} atoms")

    def build_app(self):
        app = dash.Dash(__name__)

        # Initial 3D plot (frame 0)
        frame0 = self.traj.xyz[0] * 10  # Å scale
        fig3d = go.Figure(data=go.Scatter3d(
            x=frame0[:, 0], y=frame0[:, 1], z=frame0[:, 2],
            mode='markers', marker=dict(size=2, color='blue'),
            hovertemplate='<b>Atom %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            text=[str(i) for i in range(len(frame0))]
        ))
        fig3d.update_layout(title="3D Trajectory Viewer", scene=dict(aspectmode='cube'))

        # RMSD line plot
        fig_rmsd = px.line(x=np.arange(self.n_frames), y=self.rmsd, title="RMSD over Time (Å)")

        app.layout = html.Div([
            html.H1("pyTrajViz: Interactive MD Trajectory Explorer"),
            dcc.Graph(id='rmsd-plot', figure=fig_rmsd),
            dcc.Slider(id='frame-slider', min=0, max=self.n_frames-1, value=0, step=1,
                       marks={i: str(i) for i in range(0, self.n_frames, max(1, self.n_frames//10))}),
            dcc.Graph(id='3d-view', figure=fig3d),
            html.Div(id='frame-info')
        ])

        @app.callback(
            [Output('3d-view', 'figure'), Output('frame-info', 'children')],
            [Input('frame-slider', 'value')]
        )
        def update_view(frame):
            pos = self.traj.xyz[frame] * 10
            new_fig = go.Figure(data=go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode='markers', marker=dict(size=2, color='blue'),
                hovertemplate='<b>Atom %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                text=[str(i) for i in range(len(pos))]
            ))
            new_fig.update_layout(title=f"Frame {frame} (RMSD: {self.rmsd[frame]:.3f} Å)")
            info = html.P(f"Current frame: {frame}, RMSD: {self.rmsd[frame]:.3f} Å")
            return new_fig, info

        return app

def main():
    parser = ArgumentParser(description="pyTrajViz: Run interactive MD trajectory web app")
    parser.add_argument("--traj", required=True, help="Trajectory file (.xtc)")
    parser.add_argument("--top", required=True, help="Topology file (.gro/.pdb)")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    args = parser.parse_args()

    app = TrajVizApp(args.traj, args.top).build_app()
    print(f"[pyTrajViz] Starting server at http://localhost:{args.port}")
    app.run_server(debug=True, port=args.port)

if __name__ == "__main__":
    main()
