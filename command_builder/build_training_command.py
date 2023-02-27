import pandas as pd
import json
import os
import argparse
import json
import glob
import textwrap
from pprint import pprint
import random
import pathlib

DATA_DIR = "data_subset"
RAND_SEED = 202302241120
#######################################
## Arguments
#######################################

args = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""
        Create a call to the fine tuning command
        for a set of video frames and captions.
    """)
)
args.add_argument(
    "-d", "--data_path", type=pathlib.Path,
    default=os.path.join(os.getcwd(),DATA_DIR,'random_frames'),
    help="Path to parent directory that has all folders that contain frames, e.g. data_subset/random_frames"
)

args.add_argument(
    "-c", "--captions_file", type=pathlib.Path,
    default=os.path.join(os.getcwd(),DATA_DIR,'train_val_videodatainfo.json'),
    help="path to file of captions, e.g. 'data_subset/train_val_videodatainfo.json'"
)

args.add_argument(
    "-o", "--dataframe_file", type=pathlib.Path,
    default=os.path.join(os.getcwd(),DATA_DIR,'processed_data.csv'),
    help="path to dataframe containig video-to-caption mapping"
)

args.add_argument(
    "-a", "--all-captions",
    action="store_true",
    help="""create fine tune command for ALL captions.
            If this flag is not set, a single random caption
            will be selected for each video as suggested by
            "https://huggingface.co/docs/transformers/main/en/tasks/image_captioning".
            """
)

args.add_argument(
    "-m", "--model",
    default="GIT_BASE_VATEX",
    help="""The name of the model to fine tune.
            Default is GIT_BASE_VATEX. NOTE: Other values may not work
            since pretraining script assumes vatex.
            We may want to consider GIT_BASE_MSRVTT_QA
            """
)

args.add_argument(
    "-b", "--batch_size",
    default=128, type=int,
    help="""batch size for training
            """
)

args.add_argument(
    "-e", "--epochs",
    default=2, type=int,
    help="""number of training epochs
            """
)

args = args.parse_args()

#######################################
##  get annotations
########################################

with open(args.captions_file, 'r') as file:
    json_data = json.load(file)

sentences = pd.DataFrame(json_data['sentences'])
sentences = sentences.drop("sen_id", axis=1)

# select random captions based on command line args
if not args.all_captions:
    random.seed(RAND_SEED)
    sentences = sentences.groupby('video_id').agg({'caption':list})
    sentences['caption'] = sentences.apply({'caption':random.choice})

sentences = sentences.sort_values("video_id")

#######################################
##  get frames
########################################
image_dir = args.data_path

frame_lists = [
    {'video_id': directory, 'image_files': [file for file in glob.glob(os.path.join(image_dir, directory, "*.jpg"))]}
    for directory in next(os.walk(image_dir))[1]
]

frame_table = pd.DataFrame(frame_lists)

########################################
##  Merge (inner join captions with frames)
########################################

data = pd.merge(left=frame_table, right=sentences, how="inner", on="video_id")
data.to_csv(args.dataframe_file)
base_command = "python -m generativeimage2text.finetune -p"
params = json.dumps({
    'type': 'train',
    'video_caption': args.dataframe_file,
    'model_name': args.model,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
})

command = f"{base_command} '{params}'"
#write command to .sh file
with open('runner.sh', 'w') as f:
    f.write(command)
