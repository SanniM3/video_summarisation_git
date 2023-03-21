# Additional Setup for UoE Team \& convenience scripts
The following are additional setup steps that are needed to get this working.
You should still follow all the directions in the official "Introduction" below this.

## Installation
1. Install additional requirements
```
pip install -r requirements_2.txt
```
2. Download pretrained model
```
./setup_download_model.sh
```
3. Install java 
```
sudo apt install default-jdk
```
4. GIT model setup
```
pip install -r requirements.txt
python setup.py build develop
```
## General Workflow:

### 1.  Download Data
Calling `./setup_download_data.sh` will do this for you and setup the following
directory structure

```
video_summarisation_git/data/
|
|-- category.txt ...................... # video category name to id mapping file
|
|-- train_val/ ........................ # dir for training & validation sets
|   |-- train_val_videodatainfo.json .. # annotation file
|   |-- pyscenedetect_frames/ ......... # dir for pyscenedetect sampled frames
|   |-- random_frames/ ................ # auto-generated: dir for randonmly sampled frames
|   |-- transnet_frames/ .............. # auto-generated: dir for transnet sampled frames
|   `-- videos/ ....................... # parent dir for videos (each video should have its own folder inside this dir)
|
`-- test/ ............................. # dir for test set (structure same as train_val)                             
    |-- test_videodatainfo.json ....... # annotation file
    `-- [...] 
```

### 2.  Sample Frames

#### You can download presampled frames here
Download and unzip them in `/data/train_val` or `/data/train` as appropriate.

- Training (2023-03-20)
  * pyscene https://storage.googleapis.com/mlpgit/data2023-03-20/train_val/train_pyscenedetect_frames.zip
  * transnet https://storage.googleapis.com/mlpgit/data2023-03-20/train_val/transnet_frames_2023-03-20_22-29.zip
  * random https://storage.googleapis.com/mlpgit/data2023-03-20/train_random_frames_3-31.zip
- Test (2023-03-20)
  * pyscene https://storage.googleapis.com/mlpgit/data2023-03-20/test_pyscenedetect_frames.zip
  * transnet https://storage.googleapis.com/mlpgit/data2023-03-20/test_transnet_frames.zip
  * random https://storage.googleapis.com/mlpgit/data2023-03-20/test_random_frames_3-31.zip
- out of order frames (out of date)
  * ~https://storage.googleapis.com/mlpgit/data/train_val/train_val.zip~
  * ~https://storage.googleapis.com/mlpgit/data/test/test_frames.zip~


#### Or generate them yourself with a script (will approx 40hrs w/ a k80, or 6hrs+ with a A100)

```
./setup_sample_frames.sh train # sample frames for training data
./setup_sample_frames.sh test # same for test
```

#### Or do it piecemeal by hand:

open `/setup_sample_frames.sh` to get an idea of the commands to run
for each sampling method.

Alternatively, you can look at the actual samplers in `/sampling_scripts`

### 3.  Create training csv

```
# for random frames
python command_builder/training_command.py -d data/train_val/random_frames/ -c data/train_val/train_val_videodatainfo.json

# or for transnet frames
python command_builder/training_command.py -d data/train_val/transnet_frames/ -c data/train_val/train_val_videodatainfo.json

# or for pyscenedetect frames
python command_builder/training_command.py -d data/train_val/pyscenedetect_frames/ -c data/train_val/train_val_videodatainfo.json
```

### 4.  Finetune Model or Download Already Finetuned Models

#### Download one already finetuned

##### onedrive
* [Onedrive link](https://uoe-my.sharepoint.com/:f:/r/personal/s2259199_ed_ac_uk/Documents/mlp%20project/finetuned%20models?csf=1&web=1&e=UH2PXg)

##### GCloud bucket (faster but costs $$)
* pyscenedetect
  - epochs 0-9
    * https://storage.googleapis.com/mlpgit/results/pyscene_models.zip
    * https://storage.googleapis.com/mlpgit/results/pyscene_losses.zip
  - epochs 10-19
    * https://storage.googleapis.com/mlpgit/results/pyscene_models2.zip
    * https://storage.googleapis.com/mlpgit/results/pyscene_losses2.zip

* random frames
  - (pending)
  - (pending)

* transnet
  - (pending)
  - (pending)

#### finetune your own
Do this for *ONE* selected sampling method using the following.

Alternatively you can call `./runner.sh` which should have everything you need,
and will be representative of the *last data* you called the training command builder on

```
python -m generativeimage2text.finetune -p '{
    "type": "train",
    "model_name": "GIT_BASE",
    "model_path": "model.pt",
    "batch_size": 3,
    "epochs": 20,
    "train_csv": "data/train_val/{FRAME DIRECTORY HERE}/processed_data_train.csv", # Be sure to swap out {FRAME DIRECTORY HERE} for the directory where your frames are
    "validation_csv": "data/train_val/{FRAME DIRECTORY HERE}/processed_data_validate.csv",
    "validation_annotations_json": "data/train_val/train_val_videodatainfo.json" #path to annotations file
}
``` 
### 5.  Run Inference
on test set:

```
python -m generativeimage2text.vc_inference -p "{'type': 'multi_video_inference', 'videos_csv': '', 'annotations_json_path': '', 'model_path':'./msrvtt_model_epoch1.pt', 'model_name':'GIT_BASE', 'predictions_file':None}"
```

on multiple models

```
python -m generativeimage2text.vc_inference -p "{'type': 'multi_video_inference_dir', 'videos_csv': '', 'annotations_json_path': '', 'model_dir':'./model_transnet', 'model_name':'GIT_BASE'}"
```

## Resources Created:

* [merged frames thumbnails for sampling comparison](https://uoe-my.sharepoint.com/:f:/r/personal/s2259199_ed_ac_uk/Documents/mlp%20project/frame_comparisons?csf=1&web=1&e=nOfCVJ)

* models (see section #4)

## FAQ

- Where can I find a A100 to finetune on?
  * try the netherlands region OR salt lake city US

- errors about cv2, pandas, numpy, etc.
  * make sure you've installed the second requirements file as described above

- errors about model.py when running the fine tuning script
  * make sure you've downloaded vatex as described above in the root dir of the project

-----
# below this point is the original readme

-----

# Introduction
This repo presents some example codes to reproduce some results in
[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100).

# Installation
- Install [azfuse](https://github.com/microsoft/azfuse). The tool is used to
  automatically download the data. The configuration of
  AzFuse has already been in this repo.

- Download the source code by
  ```shell
  git clone https://github.com/microsoft/GenerativeImage2Text.git
  cd GenerativeImage2Text
  ```

- Install the package
  ```shell
  pip install -r requirements.txt
  python setup.py build develop
  ```

# Inference
- Inference on a single image or multiple frames:
  ```shell
  # single image, captioning
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': 'aux_data/images/1.jpg', \
        'model_name': 'GIT_BASE', \
        'prefix': '', \
  }"
  # single image, question answering
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': 'aux_data/images/1.jpg', \
        'model_name': 'GIT_BASE_VQAv2', \
        'prefix': 'what is it?', \
  }"
  # multiple images, captioning
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': ['aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg'], \
        'model_name': 'GIT_BASE_VATEX', \
        'prefix': '', \
  }"
  # multiple images, question answering
  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': ['aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg'], \
        'model_name': 'GIT_BASE_MSRVTT_QA', \
        'prefix': 'what is it?', \
  }"
  ```
  - If `prefix` is empty, it is effectively the captioning task.
  - If `prefix` is a question, it is effectively the visual question answering task.
  - Use a list for `image_path` if it is for video. The example here is 6 identical images, only
    for a demo purpose. It should be different image frames from a video.
  - `model_name` here can be the following. Performance details can be found in the reference paper.

    | model_name          | Information                                         | Performance             |
    |---------------------|-----------------------------------------------------|------------------------ |
    | GIT_BASE            | pretrained on 4M images                             |                         |
    | GIT_BASE_COCO       | fine-tuned on COCO                                  | CIDEr: 131.4            |
    | GIT_BASE_TEXTCAPS   | fine-tuned on TextCaps for captioning               | val/CIDEr: 64.9         |
    | GIT_BASE_VQAv2      | fine-tuned on VQAv2                                 | test-dev: 72.72         |
    | GIT_BASE_TEXTVQA    | fine-tuned on TextVQA                               | val/acc: 18.81          |
    | GIT_BASE_VATEX      | fine-tuned on VATEX for captioning                  | public/test/CIDEr: 60.0 |
    | GIT_BASE_MSRVTT_QA  | fine-tuned on MSRVTT for question answering         | acc: 41.0               |
    | GIT_LARGE           | pretrained on 14M images                            |                         |
    | GIT_LARGE_COCO      | fine-tuned on COCO                                  | CIDEr: 138.5            |
    | GIT_LARGE_TEXTCAPS  | fine-tuned on TextCaps for captioning               | val/CIDEr: 106.3        |
    | GIT_LARGE_VQAv2     | fine-tuned on VQAv2                                 | test-dev: 75.51         |
    | GIT_LARGE_TEXTVQA   | fine-tuned on TextVQA                               | val/acc: 37.47          |
    | GIT_LARGE_VATEX     | fine-tuned on VATEX for captioning                  | public/test/CIDEr: 72.5 |
    | GIT_LARGE_MSRVTT_QA | fine-tuned on MSRVTT for question answering         | acc: 42.7               |

  - In the dataset of cc12m, the caption may contain some special tags to hide
    person names and the
    model might also predict such special tokens. To eliminate this issue, we
    remove these captions (around 25\% in cc12m), and re-trained the
    large-sized model. The base-sized model is not affected as cc12 is not part
    of the training data.

    | model_name          | Information                                         | Performance             |
    |---------------------|-----------------------------------------------------|------------------------ |
    | GIT_LARGE_R         | pretrained on 14M images with special tag removed   |                         |
    | GIT_LARGE_R_COCO    | fine-tuned on COCO                                  | CIDEr: 137.6            |
    | GIT_LARGE_R_TEXTCAPS| fine-tuned on TextCaps for captioning               | val/CIDEr: 105.3        |


- Inference on a [TSV](https://en.wikipedia.org/wiki/Tab-separated_values) file, which is a collection of multiple images.
  - Data format (for information only)
    - image TSV: Each row has two columns. The first is the image key; the
      second is base64-encoded jpg or png bit string.
    - caption or question tsv: Each row has two columns. The first is the image
      key; the second is a list of dictionaries in the json format. For caption TSV,
      the dictionary should contain at least the field of `'caption'`. For the
      question answering TSV, it should contain at least `question_id` and
      `question`.
  - inference on [COCO](https://cocodataset.org) Karpathy test.
      <!---
    1. Prepare the coco test TSV
       ```
       mkdir -p aux_data/raw_data
       wget http://images.cocodataset.org/zips/val2014.zip -O aux_data/raw_data/val2014.zip
       wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -O aux_data/raw_data/caption_datasets.zip
       cd aux_data/raw_data
       unzip val2014.zip
       unzip caption_datasets.zip
       python -m generativeimage2text.data_prepare -p "{'type': 'prepare_coco_test'}"
       ```
       -->
    1. Inference.
       ```shell
       # base
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/coco_caption/test.img.tsv', \
             'model_name': 'GIT_BASE_COCO', \
             'question_tsv': null, \
             'out_tsv': 'inference/GIT_BASE_COCO/coco.tsv', \
       }"
       # GIT_LARGE_COCO. If there are 8 GPUs, it can parallel by mpirun -n 8
       AZFUSE_TSV_USE_FUSE=1 mpirun -n 8 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/coco_caption/test.img.tsv', \
             'model_name': 'GIT_LARGE_COCO', \
             'question_tsv': null, \
             'out_tsv': 'inference/GIT_LARGE_COCO/coco.tsv', \
       }"
       ```
    2. Calculate the evaluation metric
       ```shell
       # base
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'evaluate_on_coco_caption', \
             'res_file': 'inference/GIT_BASE_COCO/coco.tsv', \
             'label_file': 'data/coco_caption/test.caption.tsv', \
       }"
       ```
       The CIDEr score should be 131.35 for `GIT_BASE_COCO` and  138.45 for `GIT_LARGE_COCO`.
       If you get lower score (e.g. 126 for the base model),
       the reason could be
       the misalignment of the environment, e.g. pytorch version.
    3. (optional) To exactly reproduce the number, please run the following:
       ```bash
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 \
           bash -c "mkdir -p /tmp/code \
                   && cd /tmp/code \
                   && pip install git+https://github.com/microsoft/azfuse.git \
                   && git clone https://github.com/amsword/generativeimage2text.git \
                   && cd generativeimage2text \
                   && pip install -r requirements.txt \
                   && python setup.py build develop \
                   && AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
                            'image_tsv': 'data/coco_caption/test.img.tsv', \
                            'model_name': 'GIT_BASE_COCO', \
                            'question_tsv': null, \
                            'out_tsv': 'inference/GIT_BASE_COCO/coco.tsv', \
                      }" \
                   &&  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'evaluate_on_coco_caption', \
                       'res_file': 'inference/GIT_BASE_COCO/coco.tsv', \
                       'label_file': 'data/coco_caption/test.caption.tsv', \
                       'outfile': 'inference/GIT_BASE_COCO/coco.score.json', \
                       }" \
                   && cat inference/GIT_BASE_COCO/coco.score.json \
                   "
       ```
  - Inference on [vqa](https://visualqa.org/index.html) test
    1. Inference
       ```shell
       # base model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/TaxVQAv2/test.tsv', \
             'model_name': 'GIT_BASE_VQAv2', \
             'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
             'out_tsv': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
       }"
       # GIT_LARGE_VQAv2 with 8 GPUs.
       AZFUSE_TSV_USE_FUSE=1 mpirun -n 8 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
             'image_tsv': 'data/TaxVQAv2/test.tsv', \
             'model_name': 'GIT_LARGE_VQAv2', \
             'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
             'out_tsv': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.tsv', \
       }"
       ```

    2. Convert the output tsv to the json format for submission to [evalai](https://eval.ai/web/challenges/challenge-page/830/overview)
       ```shell
       # base model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
             'predict_file': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
             'out_json': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.json', \
       }"
       # large model
       AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
             'predict_file': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.tsv', \
             'out_json': 'inference/GIT_LARGE_VQAv2/snapshot/vqav2.json', \
       }"
       ```
       Submit the file of `inference/GIT_BASE_VQAv2/snapshot/vqav2.json` to evalai
       and you should get `72.72` on `test-dev`. If it is `GIT_LARGE_VQAv2`, the accuracy is
       `75.51`.

    3. (optional) To exactly reproduce the number, you can use the
       following:
       ```shell
       # base model
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 \
           bash -c "mkdir /tmp/code \
                   && cd /tmp/code \
                   && pip install git+https://github.com/microsoft/azfuse.git \
                   && git clone https://github.com/amsword/generativeimage2text.git \
                   && cd generativeimage2text \
                   && pip install -r requirements.txt \
                   && python setup.py build develop \
                   && AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
                       'image_tsv': 'data/TaxVQAv2/test.tsv', \
                       'model_name': 'GIT_BASE_VQAv2', \
                       'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
                       'out_tsv': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
                   }" \
                   &&  AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'convert_tsv_to_vqa_json', \
                       'predict_file': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.tsv', \
                       'out_json': 'inference/GIT_BASE_VQAv2/snapshot/vqav2.json', \
                   }" \
       }"
       ```
       Note that, please modify the docker command properly so that the output
       file can be saved permanently to the host machine. It is also recommended
       to run it inside the docker container by
       ```shell
       nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 sleep infinity
       docker ps # get the docker container ID
       docker exec -it container_id /bin/bash # attach inside the docker container
       # all other commands to run the inference.
       ```

# Training
The repo shows the key code path of constructing the network
input with transformations and forward/backward. The code can be plugged into
any trainer easily. Here is the example for the base model.
- Pretraining/captioning
  ```
  python -m generativeimage2text.train -p "{'type': 'forward_backward_example', \
                  'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'], \
                  'captions': ['a couple of boats in a large body of water.', 'a view of a mountain with a tree'], \
              }"
  ```
- VQA
  ```
  python -m generativeimage2text.train -p "{'type': 'forward_backward_example', \
                  'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'], \
                  'prefixs': ['what is this?', 'how many trees?'], \
                  'captions': ['several boats in a large body of water', '1'], \
              }"
  ```


# ImageNet
## Class ID to unique readable names
- Save the file of `LOC_synset_mapping.txt` from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=LOC_synset_mapping.txt).
  under `aux_data/imagenet/`

- Convert the wordnet ID to readable names as follows
  ```python
  python -m generativeimage2text.data_prepare -p "{'type': 'generate_imagenet_unique_names'}"
  ```
  The input file is hard coded as `./aux_data/imagenet/LOC_synset_mapping.txt` and the
  output file is `./aux_data/imagenet/imagenet_unique_readable_names.txt`

# Citation
Please consider to cite the following reference if it helps.
```text
@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
```

# Acknowledgement
Part of the code is based on
[transformers](https://github.com/huggingface/transformers),
[clip](https://github.com/openai/CLIP),
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
[oscar](https://github.com/microsoft/Oscar),
[virtex](https://github.com/kdexd/virtex).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
