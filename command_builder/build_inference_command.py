"""A utility script for generating a lists of frames for GIT
"""

import os
import argparse
import json
import glob
import textwrap
from pprint import pprint

args = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    A utility script for generating a lists of frames.
    A few points on usage:
        (1) call this from the root directory of the project
        (2) set the data_path to 'sampling_scripts/out/random_frames/'

    If you'd like to run inference on *all* videos, you can do the following:
      1. redirect this output to a shell script, e.g. `$ {this script} > run_inference.sh`
      2. chmod +x that shell script, e.g. `$ chmod +x run_inference.sh`
      3. call the script, e.g. `$ ./run_inference.sh`
    ''')
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

###################################
# This is the template for the parameters to be passed to the command.
# Change this if you change the model used, etc.
###################################
for frame_list in frame_lists:
    data = json.dumps({
        'type': 'test_git_inference_single_image', # seems wrong, but this is what the instructions say it should be...
        'image_path': frame_list, 
        'model_name': 'GIT_BASE_MSRVTT_QA', # I'm not sure if this is the actual model we want, but non-QA MSRVTT doesn't seem to be an option
        'prefix': ''
    })
    print(f"AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p '{data}'")
    #pprint(data)
