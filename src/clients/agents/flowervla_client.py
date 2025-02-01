import requests
import json_numpy
from datetime import datetime
from pathlib import Path
import numpy as np

json_numpy.patch()

def from_standard_obs(std_obs, ensemble):
    cf_obs = {
        "primary_image": std_obs["primary_camera"],
        "secondary_image": std_obs["secondary_camera"],
        "ensemble": ensemble
    }

    return cf_obs

class FlowerVLAClient:
    def __init__(self, server_url: str = "http://0.0.0.0:8003", ensemble=None):
        self.server_url = server_url
        self.ensemble = ensemble

    def __call__(self, observation: dict):
        try:
            payload = from_standard_obs(observation, self.ensemble)
            response = requests.post(f"{self.server_url}/query", json=payload)
            response.raise_for_status()
            
            ret = json_numpy.loads(response.json())

            ret = ret.copy()   
            print("gripper: ", ret[..., -1])
            ret[..., -1] = 1.0 if ret[..., -1] > 0.1 else -1.0

            return ret
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None

    def reset(self, task_name: str):
        try:
            payload = {"text": task_name}
            response = requests.post(f"{self.server_url}/reset", json=payload)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None
