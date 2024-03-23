import plotly.graph_objects as go

class Plot:
    def __init__(self, title=None, cut_axis=True):
        self.fig = go.Figure()
        if title is not None:
            self.title = title
            title_length = len(title)
            if title_length < 25:
                self.font_size = 30
            elif title_length < 30:
                self.font_size = 28
            elif title_length < 35:
                self.font_size = 26
            elif title_length < 40:
                self.font_size = 24
            else:
                self.font_size = 22
        else:
            self.title = "Plot"
            self.font_size = 30

        self.tajectories = []

        self.cut_axis = cut_axis
        self.x_range = [0, 0]
        self.y_range = [0, 0]
        self.z_range = [0, 0]
        
    def compute_ranges(self):
        """
        Computes the ranges of the plot based on the trajectories.
        """
        poses = [pose for trajectory in self.tajectories for pose in trajectory]
        min_x = min([pose[0] for pose in poses])
        max_x = max([pose[0] for pose in poses])
        min_y = min([pose[1] for pose in poses])
        max_y = max([pose[1] for pose in poses])
        min_z = min([pose[2] for pose in poses])
        max_z = max([pose[2] for pose in poses])

        # r_xy = (max_x - min_x) / (max_y - min_y)
        # r_xz = (max_x - min_x) / (max_z - min_z)

        # min_y = min_y*r_xy
        # max_y = max_y*r_xy
        # min_z = min_z*r_xz
        # max_z = max_z*r_xz

        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]
        self.z_range = [min_z, max_z]

    def add_trajectory(self, trajectory, name, color, thickness=2):
        """
        Adds a trajectory to the plot.

        Parameters:
        - trajectory (list): List of 3D poses representing the trajectory.
        - name (str): Name of the trajectory.
        - color (str): Color of the trajectory line.

        Returns:
        None
        """
        self.tajectories.append(trajectory)
        self.compute_ranges()
        self.fig.add_trace(go.Scatter3d(x=[pose[0] for pose in trajectory], y=[pose[1] for pose in trajectory], z=[pose[2] for pose in trajectory], mode='lines', name=name, line=dict(color=color, width=thickness)))

    def add_points(self, point, name, color, size=5):    
        """
        Adds a point to the plot.

        Parameters:
        - point (list): 3D pose representing the point.
        - name (str): Name of the point.
        - color (str): Color of the point.

        Returns:
        None
        """
        self.fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', name=name, marker=dict(color=color, size=size)))


    def show(self):
        """
        Displays the plot.
        """
        self.fig.update_layout(
            title = self.title,
            title_x = 0.5,
            title_font = dict(size=self.font_size),
            autosize=False,
            width=1100,
            height=800,
            legend=dict(
                x=0.87,
                y=0.85,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            ),
            # updatemenus=[
            #     dict(
            #         type="buttons",
            #         direction="down",
            #         showactive=False,
            #         buttons=[
            #             dict(
            #                 args=[{"scene.camera.up": dict(x=0, y=0, z=1), "scene.camera.eye": dict(x=0.1, y=0.2, z=2)}],
            #                 label="Top View",
            #                 method="relayout"
            #             )
            #         ],
            #         x=0.87,
            #         xanchor="left",
            #         y=0.7,
            #         yanchor="top"
            #     ),
            #     dict(
            #         type="buttons",
            #         direction="down",
            #         showactive=False,
            #         buttons=[
            #             dict(
            #                 args=[{"scene.camera.up": dict(x=0, y=0, z=1), "scene.camera.eye": dict(x=-2, y=0, z=0)}],
            #                 label="Side View",
            #                 method="relayout"
            #             )
            #         ],
            #         x=0.87,
            #         xanchor="left",
            #         y=0.63,
            #         yanchor="top"
            #     ),
            #     dict(
            #         type="buttons",
            #         direction="down",
            #         showactive=False,
            #         buttons=[
            #             dict(
            #                 args=[{"scene.camera.up": dict(x=0, y=0, z=1), "scene.camera.eye": dict(x=0.01, y=-2, z=0.1)}],
            #                 label="Front View",
            #                 method="relayout"
            #             )
            #         ],
            #         x=0.87,
            #         xanchor="left",
            #         y=0.56,
            #         yanchor="top"
            #     )
            # ],
        )

        if self.cut_axis:
            self.fig.update_layout(
                scene=dict(
                    xaxis = dict(range=self.x_range),
                    yaxis = dict(range=self.y_range),
                    zaxis = dict(range=self.z_range),
                    camera=dict(
                        center=dict(x=-0.2, y=0.2, z=-0.15) 
                    ),
                    aspectmode='cube'
                )
            )
        else:
            self.fig.update_layout(
                scene=dict(
                    xaxis = dict(autorange=True),
                    yaxis = dict(autorange=True),
                    zaxis = dict(autorange=True),
                    camera=dict(
                        center=dict(x=-0.2, y=0.2, z=-0.15)
                    ),
                    aspectmode='cube'
                )
            )


        self.fig.show()

    def save(self, filename):
        """
        Saves the plot as an HTML file.

        Parameters:
        - filename (str): The name of the file to save the plot as.

        Returns:
        - None
        """
        self.fig.write_html(filename)