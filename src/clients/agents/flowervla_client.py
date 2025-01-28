# import requests
# import json_numpy
# from datetime import datetime
# from pathlib import Path
# import numpy as np

# class FlowerVLAClient:
#     def __init__(self, server_url: str = "http://0.0.0.0:8000"):
#         self.server_url = server_url

#     def __call__(self, observation: dict, instruction: str):
#         """
#         Calls the FlowerVLA server's `/act` endpoint to predict an action for a given observation and instruction.

#         :param observation: A dictionary representing the observation data.
#         :param instruction: A string with the instruction for the model.
#         :return: The predicted action as a numpy array or None if an error occurs.
#         """
#         try:
#             payload = {"observation": observation, "instruction": instruction}
#             response = requests.post(f"{self.server_url}/act", json=payload)
#             response.raise_for_status()
            
#             return json_numpy.loads(response.content)
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed: {e}")
#             return None
#         except Exception as e:
#             print(f"Error processing response: {e}")
#             return None

#     def reset(self, task_name: str):
#         """
#         Calls the FlowerVLA server's `/reset` endpoint to reset the task with a new task name.

#         :param task_name: A string representing the task description.
#         :return: Response status message or None if an error occurs.
#         """
#         try:
#             payload = {"task": task_name}
#             response = requests.post(f"{self.server_url}/reset", json=payload)
#             response.raise_for_status()
#             return response.text
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed: {e}")
#             return None
#         except Exception as e:
#             print(f"Error processing response: {e}")
#             return None
