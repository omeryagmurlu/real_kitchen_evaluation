import cv2

def map_task(key) -> str:
    """
    Map the task key to the task name

    Parameters:
    - key: task key
    """

    tasks = {
        'banana_from_right_stove_to_oven_tray': 'Move banana from right stove to oven tray',
        'pot_from_sink_to_left_stove': 'Move pot from sink to left stove',
        'open_microwave': 'Open the microwave',
        'pot_from_left_to_right_stove': 'Move pot from left to right stove',
        'banana_from_tray_to_right_stove': 'Move banana from tray to right stove',
        'pull_oven_tray': 'Pull the oven tray',
        'banana_from_right_stove_to_sink': 'Move banana from right stove to sink',
        'close_oven': 'Close the oven',
        'push_toaster_lever': 'Push down the toaster lever',
        'pot_from_right_to_left_stove': 'Move pot from right to left stove',
        'pot_from_right_stove_to_sink': 'Move pot from right stove to sink',
        'open_ice': 'Open the ice box',
        'open_oven': 'Open the oven',
        'push_oven_tray': 'Push the oven tray',
        'close_microwave': 'Close the microwave',
        'banana_from_sink_to_right_stove': 'Move banana from sink to right stove',
        'pot_from_left_stove_to_sink': 'Move pot from left stove to sink',
        'close_ice': 'Close the ice box',
        'pickup_toast_and_put_to_sink': 'Pick up toast and put it in the sink',
        'pot_from_sink_to_right_stove': 'Move pot from sink to right stove',
    }

    return tasks[key]

def resize_and_crop(
        cam_img_array, position, des_width: int = 500, des_height: int = 500
    ):
    """
    Resizes and crops an image to a square shape, centered or aligned to the left or right side.

    Arguments:
    - image_array: a NumPy array representing the image to be resized and cropped.
    - position: a string specifying the position of the crop within the image.
                Valid values are 'left', 'right', 'center', and 'center_right'.
    - size: a tuple specifying the desired size of the output image after resizing. The default value is (500, 500).

    Returns:
    - image_resized: a NumPy array representing the resized and cropped image.

    The method first calculates the size of the largest square that fits inside the original image,
    then crops the image to a square centered or aligned to the specified position.
    Finally, the cropped image is resized to the specified output size using the OpenCV library.

    Note that this method modifies the input image array in-place,
    so make a copy of the original image if you need to keep it intact.
    """


    # cam_image = np.load(cam_img_path)

    size = (des_width, des_height)
    # O Get the height and width of the image
    height, width, _ = cam_img_array.shape
    # Calculate the size of the square
    square_size = min(width, height)
    # Calculate the left, top, right, and bottom coordinates of the square
    if position == "left":
        # Crop the left side of the image
        left = 0
        top = 0
        right = square_size
        bottom = square_size
    elif position == "right":
        # Crop the right side of the image
        left = width - square_size
        top = 0
        right = width
        bottom = square_size
    elif position == "center":
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size 
        bottom = top + square_size 
    elif position == "top_center_new_lab":
        left = 550
        top = 0
        right = 1700
        bottom = top + square_size 
    elif position == "front_center_new_lab":
        left = 0
        top = 0
        right = 1700
        bottom = top + square_size
    elif position == "center_left":
        left = (width - square_size) // 2 - 30
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_far_left":
        left = (width - square_size) // 2 - 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_right":
        left = (width - square_size) // 2 + 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    else:
        raise ValueError("Invalid position. Use 'left' or 'right' or 'center'.")

    # Crop the image to create a square
    image_cropped = cam_img_array[top:bottom, left:right]

    # Resize the image to 500x500
    image_resized = cv2.resize(image_cropped, size)

    # Downscale image to 250x250
    # image_resized = cv2.resize(image_resized, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)

    return image_resized

def kitchen_dataset_obs_transforms(obs):
        """These transforms are embedded within the dataset, so EVERY policy,
        regardless of their own transforms needs to undergo these beforehand"""
        
        primary_image = obs["primary_camera"]  # [H, W, 3]
        secondary_image = obs["secondary_camera"]  # [H, W, 3]

        primary_image = resize_and_crop(primary_image, "top_center_new_lab", 224, 224)
        secondary_image = resize_and_crop(secondary_image, "front_center_new_lab", 128, 128)

        obs['primary_camera'] = primary_image
        obs['secondary_camera'] = secondary_image

        return obs