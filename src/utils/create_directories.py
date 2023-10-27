import os
from datetime import datetime

def create_timestamped_folder(base_dir):
    """
    Create a folder named with the current timestamp in the specified base directory.

    Parameters:
    - base_dir: str, the directory where the timestamped folder should be created.

    Returns:
    - folder_path: str, the path of the created folder.
    """

    # Get the current time and format it as a string
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Join the base directory with the current time to get the full path
    folder_path = os.path.join(base_dir, current_time)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    return folder_path
