import pandas as pd
import json

import os
import argparse
import json
import glob
import textwrap
from pprint import pprint
import random

args = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
     ''')
)
args.add_argument(
    "data_path", type=str,
    default='.',
    help="Path to parent directory that has all folders that contain frames"
)

args.add_argument(
    "annotations_file", type=str,
    default='.',
    help="path to file of annotations, e.g. '../train_val_videodatainfo.json'"
)

args = args.parse_args()

# get annotations
#annotations_path = "train_val_videodatainfo.json"
annotations_path = args.annotations_file

with open(annotations_path, 'r') as file:
    json_data = json.load(file)
    sentences = pd.DataFrame(json_data['sentences'])

# sentences = sentences.groupby('video_id').agg({'caption':list})

# get frames
#image_dir = "../sampling_scripts/out/random_frames/"
image_dir = args.data_path

frame_lists = [
    {'video_id': directory, 'image_files': [file for file in glob.glob(os.path.join(image_dir, directory, "*.jpg"))]}
    for directory in next(os.walk(image_dir))[1]
]

frame_table = pd.DataFrame(frame_lists)

# merge
data = pd.merge(left=frame_table, right=sentences, how="inner", on="video_id")
#data['type'] = 'forward_backward_example' # an arg the actual command line will use

base_command = "python -m generativeimage2text.finetune -p"
params = json.dumps({
    'type': 'forward_backward',
    'video_files': list(data['image_files']),
#    'model_name': 'GIT_BASE_MSRVTT_QA',
    'model_name': 'GIT_BASE_VATEX',
    'captions': list(data['caption']),
})

print(f"{base_command} '{params}'")


#### dustbin:
# TODO it's not clear how to handle multiple sentences per
# frame/video. The hugging face doc reccomends just piceking one sentence.
# the following implements that, but it might not be what we want to do.

# sentences.apply({'caption':random.choice})

# records = json.loads(data[['type', 'image_files', 'caption']].to_json(orient='records'))
