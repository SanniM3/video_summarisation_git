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

DATA_DIR = os.path.join("data","train_val")
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
    default="GIT_BASE",
    help="""The name of the model to fine tune.
            Default is GIT_BASE.
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
    help="print info level logging"
)

args.add_argument(
    "-vv", "--debug",
    action="store_true",
    help="print debug level logging"
)

args.add_argument(
    "-f", "--filter",
    type=int,
    help="only output a certain category of video. Category should be an integer"
)

args = args.parse_args()

# show/hide debugging
if args.verbose:
    logging.getLogger().setLevel(logging.INFO)
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

########################################
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

########################################
##  get categories
########################################

category_table = pd.read_table(
    pathlib.Path(os.path.join(os.getcwd(),DATA_DIR)).parent / 'category.txt',
    delimiter='\t',
    header=None,
    names=['category_name','category']
)

########################################
##  get frames
########################################

logging.debug(f"================= getting Frames =================")

image_dir = args.data_path

frame_lists = [
    {
        'video_id': directory,
        'image_files': [file for file in sorted(glob.glob(os.path.join(image_dir, directory, "*.jpg")))]
    } for directory in next(os.walk(image_dir))[1]
]
frame_table = pd.DataFrame(frame_lists)
num_videos =len(frame_lists)
num_frames = frame_table.explode('image_files')['image_files'].count()
logging.debug(f"number of directories (i.e. videos) found: {num_videos}")
logging.debug(f"number of frames found: {num_frames}")
if num_videos < 1 or num_frames < 1:
    raise Exception(f"Couldn't find any videos/frames. videos: '{num_videos}', frames: '{num_frames}'") 


########################################
##  Merge (inner join captions with frames)
########################################

data = pd.merge(left=frame_table, right=sentences, how="inner", on="video_id")
data = pd.merge(left=data, right=video_meta_data, how="inner", on="video_id")
data = pd.merge(left=data, right=category_table, how="inner", on="category")
data = data.set_index("video_id")

if args.filter:
    data = data[data['category'] == args.filter]

splits = data['split'].unique()
logging.debug(f"creating csv files for data splits: '{splits}'")
csv_files = {}
for split in splits:
    cur_split = data[ data['split'] == split ]
    csv_out = f'processed_data_{split}.csv'
    csv_out = pathlib.Path(image_dir) / csv_out
    csv_files[split] = str(csv_out)
    cur_split.to_csv(csv_out)
    logging.debug(f"  creating '{csv_out}': {'SUCCESS' if os.path.isfile(csv_out) else 'FAIL'}")
    print(f"csv for '{split}' created with {cur_split.shape[0]} rows. (Please check if this is expected)")


    # print statistical info for debugging, etc.
    logging.info(f"============= statistics for split '{split}' =============")
    logging.info(f"\nraw data:\n{cur_split}")
    logging.info(f"\ndata stats:\n{cur_split.describe()}")
    logging.info(f"\ncategory counts:\n{pd.value_counts(cur_split['category'], sort=False)}")
    logging.info(f"\nlength histogram:\n{pd.value_counts(cur_split['length'], bins=20, sort=False)}")
    length_by_category = pd.pivot_table(
        data=cur_split,
        index="category",
        aggfunc={"length":{len,"mean","std"}},
    )
    logging.info(f"\nLength statistics by video category:\n{length_by_category}")

if 'train' in csv_files.keys() and 'validate' in csv_files.keys():
    base_command = "python -m generativeimage2text.finetune -p"
    params = json.dumps({
        'type': 'train',
        'model_name': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        "train_csv": csv_files['train'],
        'model_path': 'model.pt', 
        "validation_csv": csv_files['validate'],
        "validation_annotations_json": str(args.captions_file)
    })

    command = f"{base_command} '{params}'\n"

    # write command to .sh file
    with open('runner.sh', 'w') as f:
        f.write(command)
else:
    print("CSV created, but training command not created, because dataset was not training split")
    print(f"splits found were: {splits}")