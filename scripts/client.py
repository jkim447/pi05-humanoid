from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

num_steps = 10
# create an image with random numbers of size 224 x 224 x 3 ranging from 0 to 1
img = np.random.rand(224, 224, 3).astype(np.float32)
wrist_img = np.random.rand(224, 224, 3).astype(np.float32)
state = np.random.rand(32).astype(np.float32)
task_instruction = "Pick up the red block and place it on the green block."

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    print(action_chunk)

    # Execute the actions in the environment.
    ...
