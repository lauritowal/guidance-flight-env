from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


class Plotter:
    def __init__(self, target, glide_angle_deg, bounds_radius_km, target_spawn_area_radius_km,
                 target_radius_km, aircraft_initial_position, runway_angle=90):
        self.target_position = target
        self.bounds_radius_km = bounds_radius_km
        self.target_spawn_area_radius_km = target_spawn_area_radius_km
        self.target_radius_km = target_radius_km
        self.runway_angle_deg = runway_angle
        self.aircraft_initial_position = aircraft_initial_position
        self.glide_angle_deg = glide_angle_deg

    def render_rgb_array(self, infos) -> np.array:
        xs = []
        ys = []
        track_angles = []
        rewards = []
        time_steps = []
        runway_angles = []
        runway_angle_errors = []
        runway_angle_thresholds = []
        aircraft_true_headings = []
        track_errors = []
        vertical_track_errors = []
        cross_track_errors = []
        pitches = []
        gammas = []
        alphas = []
        altitude_rates_fps = []
        altitudes = []
        altitude_errors = []
        aircraft_zs = []
        in_area = []
        winds_north_fps = []
        winds_east_fps = []
        drifts = []
        in_area_colors = []
        for info in infos:
            xs.append(info["aircraft_y"])
            ys.append(info["aircraft_x"])
            track_angles.append(info["aircraft_track_angle_deg"])
            drifts.append(info["drift_deg"])
            aircraft_true_headings.append(info["aircraft_heading_true_deg"])
            rewards.append(info["reward"])
            winds_north_fps.append(info["total_wind_north_fps"])
            winds_east_fps.append(info["total_wind_east_fps"])
            altitude_rates_fps.append(info["altitude_rate_fps"])
            runway_angle_errors.append(info["runway_angle_error"])
            runway_angle_thresholds.append(info["runway_angle_threshold_deg"])
            time_steps.append(info["simulation_time_step"])
            runway_angles.append(info["runway_angle"])
            altitudes.append(info["altitude"])
            pitches.append(np.degrees(info["pitch_rad"]))
            gammas.append(info["gamma_deg"])
            alphas.append(np.degrees(info["alpha_rad"]))
            aircraft_zs.append(info["aircraft_z"])
            altitude_errors.append(info["altitude_error"])
            track_errors.append(info["track_error"])
            vertical_track_errors.append(info["vertical_track_error"])
            cross_track_errors.append(info["cross_track_error"])
            in_area.append(info["in_area"])

            if info["in_area"] == True:
                in_area_colors.append([255, 0, 0])
            else:
                in_area_colors.append([0, 0, 255])


        # current_time_step = len(rewards)
        #
        figure = plt.figure(figsize=[10,9])
        gs = gridspec.GridSpec(4, 2, width_ratios=[2, 2])

        canvas = FigureCanvas(figure)

        ax1 = plt.subplot(gs[0])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax2 = plt.subplot(gs[2])
        ax2.set_xlabel('reward')

        ax3 = plt.subplot(gs[3])
        ax3.set_xlabel('track error')

        ax4 = plt.subplot(gs[1])
        ax4.set_axis_off()

        ax5 = plt.subplot(gs[4])
        ax5.set_xlabel('altitude (ft)')

        ax6 = plt.subplot(gs[5])
        ax6.set_xlabel('wind east & north (fps)')

        ax1.set_xlim([-self.bounds_radius_km + self.aircraft_initial_position.x,
                       self.bounds_radius_km + self.aircraft_initial_position.x])
        ax1.set_ylim([-self.bounds_radius_km + self.aircraft_initial_position.y,
                       self.bounds_radius_km + self.aircraft_initial_position.y])

        bounds = plt.Circle((self.aircraft_initial_position.x, self.aircraft_initial_position.y),
                            self.bounds_radius_km, fill=False, color='red')

        target = plt.Circle((self.target_position.x + self.aircraft_initial_position.x,
                             self.target_position.y + self.aircraft_initial_position.y),
                            self.target_radius_km, fill=False, color='green')


        target_spawn_area = plt.Circle((self.aircraft_initial_position.x, self.aircraft_initial_position.y),
                                       self.target_spawn_area_radius_km, fill=False, color='grey')

        text = plt.Text(x=0, y=0, text=f'angle error: {np.round(runway_angle_errors[-1], 2)},'
                                       f'runway_angle: {np.round(self.runway_angle_deg, 2)},'
                                       f'altitude error: {np.round(altitude_errors[-1], 2)} \n'
                                       f'wind north: {np.round(winds_north_fps[-1], 2)}, '
                                       f'wind east: {np.round(winds_east_fps[-1], 2)} \n'
                                       f'track angle: {np.round(track_errors[-1], 2)} \n'
                                       f'drift angle: {np.round(drifts[-1], 2)} \n'
                                       f'rewards {np.round(np.sum(rewards), 2)}')

        ax1.set_aspect(1)
        ax1.add_artist(bounds)
        ax1.add_artist(target)
        ax1.add_artist(target_spawn_area)

        ax4.add_artist(text)

        # See https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-scatter-plot-matplotlib
        ax1.scatter(xs, ys, c=np.array(in_area)/255.0, s=0.1)

        ax2.plot(time_steps, rewards, c='red')

        ax3.plot(time_steps, track_errors)
        ax3.plot(time_steps, cross_track_errors)
        ax3.plot(time_steps, vertical_track_errors)
        ax3.legend(["track", "cross", "vertical"])

        ax5.plot(time_steps, altitudes)

        ax6.plot(time_steps, winds_east_fps)
        ax6.plot(time_steps, winds_north_fps)
        ax6.legend(["wind east", "wind north"])

        canvas.draw()
        rendered = np.array(canvas.renderer.buffer_rgba())
        plt.close('all')
        return rendered

    def plot_html(self, infos, path="./htmls/test.html"):
        xs = []
        ys = []
        track_angles = []
        rewards = []
        time_steps = []
        runway_angles = []
        runway_angle_errors = []
        runway_angle_thresholds = []
        aircraft_true_headings = []
        track_errors = []
        altitude_rates_fps = []
        altitudes = []
        altitude_errors = []
        aircraft_zs = []
        in_area = []
        in_area_colors = []
        for info in infos:
            xs.append(info["aircraft_y"])
            ys.append(info["aircraft_x"])
            track_angles.append(info["aircraft_track_angle_deg"])
            aircraft_true_headings.append(info["aircraft_heading_true_deg"])
            rewards.append(info["reward"])
            altitude_rates_fps.append(info["altitude_rate_fps"])
            runway_angle_errors.append(info["runway_angle_error"])
            runway_angle_thresholds.append(info["runway_angle_threshold_deg"])
            time_steps.append(info["simulation_time_step"])
            runway_angles.append(info["runway_angle"])
            altitudes.append(info["altitude"])
            aircraft_zs.append(info["aircraft_z"])
            altitude_errors.append(info["altitude_error"])
            track_errors.append(info["track_error"])
            in_area.append(info["in_area"])
            if info["in_area"] == True:
                in_area_colors.append([255, 0, 0])
            else:
                in_area_colors.append([0, 0, 255])

        fig = make_subplots(
            rows=3,
            cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.6, 0.4, 0.6],
            specs=[[{"type": "scatter3d", "rowspan": 2}, {"type": "scatter"}],
                   [None, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=aircraft_zs,
                customdata=time_steps,
                hovertemplate='x: %{x}' + '<br>y: %{y}<br>' + 'altitude: %{z}<br>' + 'time: %{customdata} s<br>',
                mode='markers',
                marker=dict(
                    size=2,
                    color=aircraft_zs,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=1
                )
            ),
            row=1, col=[1,2]
        )

        fig.write_html(path)
