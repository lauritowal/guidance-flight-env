from ray.rllib.agents.callbacks import DefaultCallbacks
from PIL import Image
from PIL import ImageDraw
import numpy as np

import os
dirname = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(dirname, 'images')

episode_counter = 0
episode_counter_file = 0


class CustomCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict):
        if result["episode_reward_mean"] > -300:
            phase = 4
        elif result["episode_reward_mean"] > -600:
            phase = 3
        elif result["episode_reward_mean"] > -750:
            phase = 2
        elif result["episode_reward_mean"] > -900:
            phase = 1
        else:
            phase = 0

        print(f"set phase to: {phase}")
        trainer.workers.foreach_worker(lambda ev: ev.foreach_env(lambda env: env.set_phase(phase)))


    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        global episode_counter
        global episode_counter_file

        print("Episode done:", episode_counter)
        envs = base_env.get_unwrapped()
        info = episode.last_info_for()

        for env in envs:
            if env.steps_left < 1 or env.done:
                rgb_array: np.array = env.render(mode="rgb_array")
                image: Image = Image.fromarray(rgb_array)
                ImageDraw.Draw(image).text((0, 0), f'episode: {episode_counter}', (0, 0, 0))

                path = ""
                if info["is_aircraft_out_of_bounds"]:
                    path = f'{IMAGE_PATH}/episode_{episode_counter_file}_bounds'
                elif info["is_aircraft_at_target"]:
                    if info["is_heading_correct"]:
                        path = f'{IMAGE_PATH}/episode_{episode_counter_file}_heading'
                    else:
                        path = f'{IMAGE_PATH}/episode_{episode_counter_file}_target'
                elif info["is_on_track"]:
                    path = f'{IMAGE_PATH}/episode_{episode_counter_file}_track'
                else:
                    path = f'{IMAGE_PATH}/episode_{episode_counter_file}_other'

                image.save(f'{path}.png')
                # env.render_html(f'{path}.html')

        #def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        if "num_aircraft_out_of_bounds_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_out_of_bounds_metric"] = 0
        if "runway_angle_error" not in episode.custom_metrics:
            episode.custom_metrics["runway_angle_error"] = 0
        if "num_on_track" not in episode.custom_metrics:
            episode.custom_metrics["num_on_track"] = 0
        if "success" not in episode.custom_metrics:
            episode.custom_metrics["success"] = 0
        if "altitude_error" not in episode.custom_metrics:
            episode.custom_metrics["altitude_error"] = 0
        if "num_aircraft_at_target_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_at_target_metric"] = 0
        if "num_correct_heading_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_correct_heading_metric"] = 0
        if "num_aircraft_not_at_target_neither_out_of_bounds_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_not_at_target_neither_out_of_bounds_metric"] = 0
        if "num_aircraft_altitude_to_low" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_altitude_to_low"] = 0
        if "num_distance_to_target" not in episode.custom_metrics:
            episode.custom_metrics["num_distance_to_target"] = 0
        if "num_distance_to_target_global" not in episode.custom_metrics:
            episode.custom_metrics["num_distance_to_target_global"] = 0
        if "num_gamma_deg" not in episode.custom_metrics:
            episode.custom_metrics["num_gamma"] = 0
        if "cross_track_error_at_end" not in episode.custom_metrics:
            episode.custom_metrics["cross_track_error_at_end"] = 0
        if "vertical_track_error_at_end" not in episode.custom_metrics:
            episode.custom_metrics["vertical_track_error_at_end"] = 0
        if "num_gamma_deg" not in episode.custom_metrics:
            episode.custom_metrics["num_gamma"] = 0
        if info["is_aircraft_out_of_bounds"]:
            episode.custom_metrics["num_aircraft_out_of_bounds_metric"] += 1
        if info["is_aircraft_altitude_to_low"]:
            episode.custom_metrics["num_aircraft_altitude_to_low"] += 1

        episode.custom_metrics["num_distance_to_target_global"] = abs(info["distance_to_target"])

        ### Count on at target only / on track

        if info["is_aircraft_at_target"]:
            episode.custom_metrics["num_aircraft_at_target_metric"] += 1

        if info["is_heading_correct"]:
            episode.custom_metrics["num_correct_heading_metric"] += 1

        if info["is_on_track"]:
            episode.custom_metrics["num_on_track"] += 1

        if info["is_aircraft_at_target"] or info["is_on_track"]:
            episode.custom_metrics["success"] += 1
            episode.custom_metrics["runway_angle_error"] = abs(info["runway_angle_error"])
            episode.custom_metrics["altitude_error"] = abs(info["altitude_error"])
            episode.custom_metrics["num_distance_to_target"] = abs(info["distance_to_target"])
            episode.custom_metrics["num_gamma_deg"] = abs(info["gamma_deg"]) # Gleitwinkel

        if not info["is_aircraft_at_target"] and not info["is_aircraft_out_of_bounds"]:
            episode.custom_metrics["num_aircraft_not_at_target_neither_out_of_bounds_metric"] += 1

        episode_counter += 1
        episode_counter_file += 1

        if episode_counter_file > 10:
            episode_counter_file = 0
