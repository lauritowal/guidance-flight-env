import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PIL import ImageDraw
import numpy as np
from PIL import Image


import plotly.graph_objects as go

from plotly.subplots import make_subplots
import numpy as np
import io
from PIL import Image
from matplotlib import pyplot as plt
from celluloid import Camera
import imageio
import io
import base64
from IPython.display import HTML
from matplotlib import gridspec


class MapPlotter:
    def __init__(self, target, glide_angle_deg, bounds_radius_km, localizer_position, localizer_glide_position, localizer_perpendicular_position,
                 example_position, target_spawn_area_radius_km, target_radius_km, aircraft_initial_position, runway_angle=90, offset=0):
        self.target_position = target
        self.localizer_position = localizer_position
        self.localizer_glide_position = localizer_glide_position
        self.localizer_perpendicular_position = localizer_perpendicular_position
        self.example_position = example_position
        self.bounds_radius_km = bounds_radius_km
        self.target_spawn_area_radius_km = target_spawn_area_radius_km
        self.target_radius_km = target_radius_km
        self.runway_angle_deg = runway_angle
        self.aircraft_initial_position = aircraft_initial_position
        self.glide_angle_deg = glide_angle_deg


    def convert2gif(self, images, file_name, interval: int = 50):
        imageio.mimsave(f'{file_name}.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=interval)

    @classmethod
    def convert2video(self, images: np.array, file_name: str, interval: int = 50):
        fig = plt.figure()
        camera = Camera(fig)
        episode_count = 0
        for image in images:
            plt.imshow(image)
            plt.figtext(0.5, 0.01, f"episode {episode_count}", ha="center", fontsize=18,
                        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
            camera.snap()
            episode_count += 1

        animation = camera.animate(interval=interval)
        file = f'{file_name}.mp4'
        animation.save(file)
        print("save to file", file)

        return animation

    @classmethod
    def save_images(self, images: np.array, infos, path: str = "./images"):
        episode_count = 0
        for image_array in images:
            rescaled = (255.0 / image_array.max() * (image_array - image_array.min())).astype(np.uint8)

            image = Image.fromarray(rescaled)
            ImageDraw.Draw(image).text((0, 0), f'episode: {episode_count}', (0, 0, 0))

            info = infos[episode_count]
            if info["is_aircraft_out_of_bounds"]:
                type = f'bounds'
            elif info["is_aircraft_at_target"]:
                if info["is_heading_correct"]:
                    type = f'heading'
                else:
                    type = f'target'
            elif info["is_on_track"]:
                type = f'track'
            else:
                type = f'other'
            image = image.convert('RGB')
            image.save(f"{path}/episode_{episode_count}_{type}.png") # , resolution=6000, quality=100

            episode_count += 1


    @classmethod
    def play_video(self, filename):
        video = io.open(filename, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" controls>
                        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                     </video>'''.format(encoded.decode('ascii')))

    def render(self, infos) -> np.array:
        xs = []
        ys = []
        track_angles = []
        rewards = []
        time_steps = []
        runway_angles = []
        runway_angle_errors = []
        runway_angle_thresholds = []
        aircraft_true_heading = []
        track_errors = []
        vertical_track_errors = []
        cross_track_errors = []
        pitches = []
        speeds = []
        gammas = []
        alphas = []
        altitude_rates_fps = []
        altitudes = []
        altitude_errors = []
        aircraft_zs = []
        in_area = []
        in_area_colors = []
        for info in infos:
            xs.append(info["aircraft_long_deg"])
            ys.append(info["aircraft_lat_deg"])
            track_angles.append(info["aircraft_track_angle_deg"])
            aircraft_true_heading.append(info["aircraft_heading_true_deg"])
            rewards.append(info["reward"])
            altitude_rates_fps.append(info["altitude_rate_fps"])
            runway_angle_errors.append(info["runway_angle_error"])
            runway_angle_thresholds.append(info["runway_angle_threshold_deg"])
            speeds.append(info["true_airspeed"])
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
        gs = gridspec.GridSpec(3, 2, width_ratios=[2, 2])

        canvas = FigureCanvas(figure)

        ax1 = plt.subplot(gs[0])
        ax1.set_xlabel('long')
        ax1.set_ylabel('lat')

        ax2 = plt.subplot(gs[2])
        ax2.set_xlabel('reward')

        ax3 = plt.subplot(gs[3])
        ax3.set_xlabel('track error')

        ax4 = plt.subplot(gs[1])
        ax4.set_axis_off()

        ax5 = plt.subplot(gs[4])
        ax5.set_xlabel('altitude (ft)')

        ax6 = plt.subplot(gs[5])
        ax6.set_xlabel('gammas (deg)')

        # ax7 = plt.subplot(gs[6])
        # ax7.set_xlabel('pitch (deg)')
        #
        # ax8 = plt.subplot(gs[7])
        # ax8.set_xlabel('alpha (deg)')


        ax1.set_xlim([-self.bounds_radius_km + self.aircraft_initial_position.x,
                       self.bounds_radius_km + self.aircraft_initial_position.x])
        ax1.set_ylim([-self.bounds_radius_km + self.aircraft_initial_position.y,
                       self.bounds_radius_km + self.aircraft_initial_position.y])

        bounds = plt.Circle((self.aircraft_initial_position.x, self.aircraft_initial_position.y),
                            self.bounds_radius_km, fill=False, color='red')

        target = plt.Circle((self.target_position.x + self.aircraft_initial_position.x,
                             self.target_position.y + self.aircraft_initial_position.y),
                            self.target_radius_km, fill=False, color='green')


        localizer = plt.Circle((self.localizer_position.x, self.localizer_position.y), radius=0.1, fill=True)
        localizer_perpendicular = plt.Circle((self.localizer_perpendicular_position.x, self.localizer_perpendicular_position.y), radius=0.1, fill=True, color="orange")
        example = plt.Circle((self.example_position.x, self.example_position.y), radius=0.1, color="red", fill=True)

        line = plt.Line2D(xdata=[self.localizer_position.x, self.target_position.x],
                          ydata=[self.localizer_position.y, self.target_position.y], color="grey")

        line_2 = plt.Line2D(xdata=[self.localizer_perpendicular_position.x, self.localizer_position.x],
                          ydata=[self.localizer_perpendicular_position.y, self.localizer_position.y], color="orange")

        target_spawn_area = plt.Circle((self.aircraft_initial_position.x, self.aircraft_initial_position.y),
                                       self.target_spawn_area_radius_km, fill=False, color='grey')

        text = plt.Text(x=0, y=0, text=f'angle error: {np.round(runway_angle_errors[-1], 2)} \n'
                                       f'runway_angle: {np.round(self.runway_angle_deg, 2)} \n'
                                       f'tas (fps): {np.round(speeds[-1], 2)} \n'
                                       f'altitude error: {np.round(altitude_errors[-1], 2)} \n'
                                       f'aircraft_z: {np.round(aircraft_zs[-1], 2)} \n'
                                       f'target_z: {np.round(self.target_position.z, 2)} \n'
                                       f'altitude_rates_fps: {np.round(altitude_rates_fps[-1], 2)} \n'
                                       f'altitude_rates_fpm: {np.round(altitude_rates_fps[-1] * 60, 2)} \n'
                                       f'rewards {np.round(np.sum(rewards), 2)}')

        ax1.set_aspect(1)
        ax1.add_artist(bounds)
        # ax1.add_artist(localizer)
        # ax1.add_artist(localizer_perpendicular)
        # ax1.add_artist(example)
        ax1.add_artist(target)
        ax1.add_artist(line)
        # ax1.add_artist(line_2)
        ax1.add_artist(target_spawn_area)
        # See https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-scatter-plot-matplotlib
        ax1.scatter(xs, ys, c=np.array(in_area_colors)/255.0, s=0.1)

        ax2.plot(time_steps, rewards, c='red')

        # ax2.plot(time_steps, altitude_rates_fps, c='red')
        # ax2.legend(["altitude_rates_fps"])

        ax3.plot(time_steps, track_errors)
        ax3.plot(time_steps, cross_track_errors)
        ax3.plot(time_steps, vertical_track_errors)
        ax3.legend(["track", "cross", "vertical"])

        ax5.plot(time_steps, altitudes)
        ax6.plot(time_steps, gammas)
        ax6.plot(time_steps, pitches)
        ax6.plot(time_steps, alphas)
        ax6.legend(["gamma", "pitch", "alpha"])


        #### extra overwrite ####
        # figure2 = plt.figure(figsize=[10,9])
        # a = plt.subplot()
        # canvas = FigureCanvas(figure2)
        #
        # a.set_xlabel('x')
        # a.set_ylabel('y')
        #
        #
        # a.set_xlim([-self.bounds_radius_km + self.aircraft_initial_position.x,
        #                self.bounds_radius_km + self.aircraft_initial_position.x])
        # a.set_ylim([-self.bounds_radius_km + self.aircraft_initial_position.y,
        #                self.bounds_radius_km + self.aircraft_initial_position.y])
        #
        #
        # a.set_aspect(1)
        # a.add_artist(bounds)
        # a.add_artist(target)
        # a.add_artist(line)
        # a.add_artist(target_spawn_area)
        # a.scatter(xs, ys, c=np.array(in_area_colors)/255.0, s=0.1, zorder=100)

        canvas.draw()
        rendered = np.array(canvas.renderer.buffer_rgba())
        plt.close('all')
        return rendered

    def _fig2array(self, fig):
        img_bytes = fig.to_image(format="png")
        buf = io.BytesIO(img_bytes)
        img = Image.open(buf)
        return np.asarray(img)

    @classmethod
    def plot_2D(self, x, y, title):
        fig = go.Figure(
            data=[],
            layout=go.Layout(title=go.layout.Title(text=title))
        )
        fig.add_trace(go.Scatter(x=x, y=y))

        fig.update_layout()
        fig.show()

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
        pitches = []
        gammas = []
        alphas = []
        altitude_rates_fps = []
        altitudes = []
        altitude_errors = []
        aircraft_zs = []
        in_area = []
        in_area_colors = []
        for info in infos:
            xs.append(info["aircraft_long_deg"])
            ys.append(info["aircraft_lat_deg"])
            track_angles.append(info["aircraft_track_angle_deg"])
            aircraft_true_headings.append(info["aircraft_heading_true_deg"])
            rewards.append(info["reward"])
            altitude_rates_fps.append(info["altitude_rate_fps"])
            runway_angle_errors.append(info["runway_angle_error"])
            runway_angle_thresholds.append(info["runway_angle_threshold_deg"])
            time_steps.append(info["simulation_time_step"])
            runway_angles.append(info["runway_angle"])
            altitudes.append(info["altitude"])
            pitches.append(np.rad2deg(info["pitch_rad"]))
            gammas.append(info["gamma_deg"])
            alphas.append(np.degrees(info["alpha_rad"]))
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

        # calculate direction

        # TODO:  information about angle
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

        zs = [self.target_position.z, self.localizer_position.z]
        fig.add_trace(
            go.Scatter3d(
                x=[self.target_position.x, self.localizer_position.x],
                y=[self.target_position.y, self.localizer_position.y],
                z=zs,
                marker=dict(
                    color=zs,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=1
                ),
                line=dict(
                    color='green',
                    width=2
                )
            ),
            row=1, col=1
        )

        zs = [self.target_position.z, self.localizer_glide_position.z]
        fig.add_trace(
            go.Scatter3d(
                x=[self.target_position.x, self.localizer_glide_position.x],
                y=[self.target_position.y, self.localizer_glide_position.y],
                z=zs,
                name="glide",
                marker=dict(
                    color=zs,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=1
                ),
                line=dict(
                    color='red',
                    width=2
                )
            ),
            row=1, col=1
        )


        # fig.add_trace(
        #     go.Scatter3d(
        #         name="example",
        #         x=[self.example_position.x],
        #         y=[self.example_position.y],
        #         z=[self.example_position.z],
        #     ),
        #     row=1, col=1
        # )


        fig.write_html(path)
