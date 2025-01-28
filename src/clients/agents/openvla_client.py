import numpy as np
import requests
import json_numpy


class OpenVLAClient:
    def __init__(self):
        self.server_url = "http://0.0.0.0:8000/act"

    def __call__(self, image: np.ndarray, instruction: str):
        """
        Calls the OpenVLA server's `/act` endpoint to predict an action for a given image and instruction.

        :param image: A numpy array representing the image (e.g., shape (256, 256, 3)).
        :param instruction: A string with the instruction for the model.
        :param server_url: The URL of the OpenVLA API server.
        :return: The predicted action as a numpy array or None if an error occurs.
        """
        try:
            payload = {"image": image, "instruction": instruction}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
            
            return json_numpy.loads(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None
        
    def reset(self):
        pass