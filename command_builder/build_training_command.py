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
import logging

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
    default=3, type=int,
    help="""batch size for training
            """
)

args.add_argument(
    "-e", "--epochs",
    default=2, type=int,
    help="""number of training epochs
            """
)

args.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="print debugging output"
)

args.add_argument(
    "-f", "--filter",
    type=int,
    help="only output a certain category of video. Category should be an integer"
)

args = args.parse_args()

# show/hide debugging
if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

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

# get video meta data
video_meta_data = pd.DataFrame(json_data['videos'])
video_meta_data.drop(['url'],axis=1)
video_meta_data['length'] = video_meta_data['end time'] - video_meta_data['start time']
video_meta_data['category'] = video_meta_data['category'].astype(int)

category_table = pd.read_table(
    os.path.join(os.getcwd(),DATA_DIR,'category.txt'),
    delimiter='\t',
    header=0,
    names=['category_name','category']
)

#######################################
##  get frames
########################################
image_dir = args.data_path

frame_lists = [
    {
        'video_id': directory,
        'image_files': [file for file in sorted(glob.glob(os.path.join(image_dir, directory, "*.jpg")))]
    } for directory in next(os.walk(image_dir))[1]
]

frame_table = pd.DataFrame(frame_lists)

########################################
##  Merge (inner join captions with frames)
########################################

data = pd.merge(left=frame_table, right=sentences, how="inner", on="video_id")
data = pd.merge(left=data, right=video_meta_data, how="inner", on="video_id")
data = pd.merge(left=data, right=category_table, how="inner", on="category")
data = data.set_index("video_id")

if args.filter:
    data = data[data['category'] == args.filter]

for split in data['split'].unique():
    cur_split = data[ data['split'] == split ]
    cur_split.to_csv(f'processed_data_{split}.csv')

    # print statistical info for debugging, etc.
    logging.debug(f"============= statistics for split '{split}' =============")
    logging.debug(f"\nraw data:\n{cur_split}")
    logging.debug(f"\ndata stats:\n{cur_split.describe()}")
    logging.debug(f"\ncategory counts:\n{pd.value_counts(cur_split['category'], sort=False)}")
    logging.debug(f"\nlength histogram:\n{pd.value_counts(cur_split['length'], bins=20, sort=False)}")
    length_by_category = pd.pivot_table(
        data=cur_split,
        index="category",
        aggfunc={"length":{len,"mean","std"}},
    )
    logging.debug(f"\nLength statistics by video category:\n{length_by_category}")

base_command = "python -m generativeimage2text.finetune -p"
params = json.dumps({
    'type': 'train',
    'model_name': args.model,
    'batch_size': args.batch_size,
    'epochs': args.epochs
})

command = f"{base_command} '{params}'"

# write command to .sh file
with open('runner.sh', 'w') as f:
    f.write(command)


#python ./video_summarisation_git/command_builder/build_training_command.py -d ./video_summarisation_git/random_train_frames/ -c ./train_val_videodatainfo.json -m GIT_BASE_VATEX -e 2 -b 128 -o processed_data.csv
