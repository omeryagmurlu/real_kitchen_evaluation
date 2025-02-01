# eval.py

from datetime import datetime
import os
import random
import logging
from pathlib import Path
from time import sleep

import hydra
import numpy as np
import imageio
import optree

from omegaconf import DictConfig

from src.kitchen import kitchen_dataset_obs_transforms
from src.utils.non_blocking_keypress import NonBlockingKeyPress

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationApp:
    def __init__(self, agent, env, tasks, task_map=None):
        # Create output folder
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        self.tasks = tasks

        # Create environment
        self.env = env
        assert self.env.connect()
        self.latest_obs, reset_info = self.env.reset()

        if task_map is not None:
            self.task_map = task_map
        else:
            self.task_map = lambda x: x

        self.agent = agent

    def _reset_robot(self):
        self.latest_obs, _ = self.env.reset()
        logger.info(f"Reset the robot")

    def _reset(self, reset_robot=True):

        # Reset robot
        if reset_robot:
            self._reset_robot()
        self.latest_obs_timestamp = datetime.now()

        logger.info(f"Reset the agent")
        self.agent.reset(self.instruction)

        # Reset storage
        self.obs_timestamps = []
        self.action_timestamps = []
        self.joint_positions = []
        self.joint_velocities = []
        self.ee_positions = []
        self.ee_velocities = []
        self.gripper_widths = []
        self.actions = []

        self.images = {
            "primary_camera": [],
            "secondary_camera": []
        }

    def _inference_step(self):

        assert self.instruction is not None

        std_obs = self.env.to_standardized_obs(self.latest_obs)
        tr_obs = kitchen_dataset_obs_transforms(std_obs)
        self.tr_obs = tr_obs

        raw_actions = self.agent(tr_obs)

        # Record data
        self.latest_action = raw_actions
        self.latest_action_timestamp = datetime.now()

        # 4) Postprocess (de-normalize, handle orientation, sticky gripper, etc.)
        processed_actions = self.env.from_standardized_action(raw_actions)

        # 5) Step the environment
        self.latest_obs, reward, done, truncated, info = self.env.step(processed_actions)

        # Record time
        self.latest_obs_timestamp = datetime.now()

    def _init_storage(self, task_name: str, subtask_number: int, task_i: int, subtask_name: str):

        # Create task and subtask folder
        subtask_dir = self.output_dir / task_name / str(task_i + 200) / (subtask_name + "_" + str(subtask_number))
        subtask_dir.mkdir(parents=True, exist_ok=True)
        self.subtask_dir = subtask_dir

        # Clean collected data
        self.observations = {}

    def _save_step(self):

        # Save timestamps
        self.obs_timestamps.append(self.latest_obs_timestamp)
        self.action_timestamps.append(self.latest_action_timestamp)

        # Save trajectories with values directly from the robot
        self.joint_positions.append(self.latest_obs["joint_pos"])
        # self.joint_velocities.append(self.latest_obs["joint_vel"])
        # self.ee_positions.append(self.latest_obs["ee_pos"])
        # self.ee_velocities.append(self.latest_obs["ee_vel"])
        # self.gripper_widths.append(self.latest_obs["gripper_width"])

        # Save the model's output
        self.actions.append(self.latest_action)

        # Save video
        self.images["primary_camera"].append(self.tr_obs["primary_camera"])
        self.images["secondary_camera"].append(self.tr_obs["secondary_camera"])

    def _save_subtask(self, success):

        assert self.subtask_dir.exists()

        # Store timestamps
        obs_timestamps = np.vstack([ts.isoformat() for ts in self.obs_timestamps])
        np.save(self.subtask_dir / "obs_timestamps.npy", obs_timestamps)
        action_timestamps = np.vstack([ts.isoformat() for ts in self.action_timestamps])
        np.save(self.subtask_dir / "action_timestamps.npy", action_timestamps)

        # Store trajectories with values directly from the robot
        joint_positions = np.vstack(self.joint_positions)
        np.save(self.subtask_dir / "joint_positions.npy", joint_positions)
        # joint_velocities = np.vstack(self.joint_velocities)
        # np.save(self.subtask_dir / "joint_velocities.npy", joint_velocities)
        # ee_positions = np.vstack(self.ee_positions)
        # np.save(self.subtask_dir / "ee_positions.npy", ee_positions)
        # ee_velocities = np.vstack(self.ee_velocities)
        # np.save(self.subtask_dir / "ee_velocities.npy", ee_velocities)
        # gripper_widths = np.vstack(self.gripper_widths)
        # np.save(self.subtask_dir / "gripper_widths.npy", gripper_widths)

        # Store the model's output
        actions = np.vstack(self.actions)
        np.save(self.subtask_dir / "actions.npy", actions)

        for camera in self.images:
            vid = np.stack(self.images[camera], axis=0)
            imageio.mimsave(self.subtask_dir / f"{camera}.mp4", vid, fps=10)

        (self.subtask_dir / ('success' if success else 'failure')).touch()

    def run_evaluation(self):

        # Run tasks
        for task_i, task in enumerate(self.tasks):
            if not (isinstance(task, DictConfig) or isinstance(task, dict)):
                task = dict(
                    name=task,
                    subtasks=[task]
                )

            self._reset_robot()

            # Ask if [p]roceed, [s]kip or [q]uit the task
            print("")
            print(f"📋 The current task is: {task['name']}")
            print(f"Do you want [p]roceed, [s]kip, [r]eset again or [q]uit?")
            key = input()
            while key != "p" and key != "s" and key != "q":
                if key == "r":
                    print("Resetting the robot again.")
                    self._reset_robot()
                else:
                    sleep(0.3)
                print("Please enter a command.")
                key = input()

            if key == "s":
                continue

            elif key == "q":
                break
            logger.info(f"Running task: {task['name']}")

            for i, subtask in enumerate(task["subtasks"]):

                # Ask if [p]roceed, [s]kip or [q]uit the subtask
                print("")
                print(f"📜 The current subtask is: '{subtask}'")
                print(f"Do you want [p]roceed, [s]kip or [q]uit?")
                key = input()
                while key != "p" and key != "s" and key != "q":
                    sleep(0.3)
                    print("Please enter a valid command")
                    key = input()
                if key == "s":
                    continue
                elif key == "q":
                    break

                # Set instruction
                self.instruction = self.task_map(subtask)
                logger.info(f"Running subtask instruction: '{subtask}'")

                # Init subtask output folder
                self._init_storage(task["name"], i, task_i, subtask)

                # Reset everything
                self._reset(reset_robot=False)

                # Loop until [s]uccess or [f]ail
                print("")
                print("📌 If the robot completed the subtask successfully press [s] otherwise [f]")
                print("")
                step = 0
                success = None
                with NonBlockingKeyPress() as kp:
                    while success is None:

                        self._inference_step()
                        step += 1

                        # Logging
                        if step % 10 == 0:
                            logger.info(f"Step {step}")

                        # Check if user has decided outcome
                        key = kp.get_data()
                        if key == "s":
                            success = True
                        elif key == "f":
                            success = False

                        # Save current state
                        self._save_step()

                # Print outcome
                if success:
                    logger.info(f"Successfully finished the subtask '{subtask}'")
                else:
                    logger.info(f"Failed finishing the subtask '{subtask}'")
                logger.info(f"The execution took: {self.obs_timestamps[-1] - self.obs_timestamps[0]}")

                # Store results
                self._save_subtask(success)

            # Task summary
            logger.info("Task evaluation completed")

        # Close connection to environment
        self.env.close()


@hydra.main(config_path="config", config_name="kitchen_eval")
def main(cfg):
    """Main evaluation entry point."""

    app = hydra.utils.instantiate(cfg)
    app.run_evaluation()

    # TODO Calculate statistics?


if __name__ == "__main__":
    main()
