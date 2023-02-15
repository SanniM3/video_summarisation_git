"""A utility script for generating a lists of frames for GIT
"""

import os
import argparse
import json
import glob
from pprint import pprint

args = argparse.ArgumentParser(
    description="""
    A utility script for generating a lists of frames
    """
)
args.add_argument(
    "data_path", type=str,
    default='.',
    help="Path to parent directory that has all folders that contain frames"
)

args = args.parse_args()

frame_lists  = [
    sorted([file for file in glob.glob(os.path.join(args.data_path, directory, "*.jpg"))])
    for directory in next(os.walk(args.data_path))[1]
]

for frame_list in frame_lists:
    data = json.dumps({
        'type': 'test_git_inference_single_image',
        'image_path': frame_list, 
        'model_name': 'GIT_BASE_VATEX',
        'prefix': ''
    })
    #print(f"AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p '{data}'")
    pprint(data)
