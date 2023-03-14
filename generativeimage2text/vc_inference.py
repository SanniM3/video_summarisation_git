from .common import Config
import json
import re
import pandas as pd
from ast import literal_eval
import os.path as op
from .common import qd_tqdm as tqdm
from .common import json_dump
from .common import pilimg_from_base64
from .common import get_mpi_rank, get_mpi_size, get_mpi_local_rank

from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File

from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load, recursive_to_device, resize_2d_pos_embed
from .torch_common import load_state_dict
from .process_image import load_image_by_pil
from .layers.CLIP import clip
from .layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
from .layers.decoder import CaptioningModel
from .process_image import load_image_by_pil
from .data_layer.transform import RenameKey, SelectTransform
from .data_layer.transform import ImageTransform2Dict
from .data_layer.transform import get_inception_train_transform
from .data_layer.builder import collate_fn
from .model import get_git_model

from .metrics.eval import EvalCap
import os


class MinMaxResizeForTest(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self):
        return 'MinMaxResizeForTest({}, {})'.format(
            self.min_size, self.max_size)

    def __call__(self, img):
        size = self.get_size(img.size)
        import torchvision.transforms.functional as F
        image = F.resize(img, size, interpolation=PIL.Image.BICUBIC)
        return image


def test_git_inference_single_image(image_path, model_name, prefix):
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if isinstance(image_path, str):
        image_path = [image_path]
    # if it is more than 1 image, it is normally a video with multiple image
    # frames
    img = [load_image_by_pil(i) for i in image_path]

    transforms = get_image_transform(param)
    img = [transforms(i) for i in img]

    # model
    model = get_git_model(tokenizer, param)
    # pretrained = f'output/{model_name}/snapshot/model.pt'
    pretrained = 'model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    model.cuda()
    model.eval()
    img = [i.unsqueeze(0).cuda() for i in img]

    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    with torch.no_grad():
        result = model({
            'image': img,
            'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
        })
    cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
    logging.info('output: {}'.format(cap))

def get_image_transform(param):
    crop_size = param.get('test_crop_size', 224)
    if 'test_respect_ratio_max' in param:
        trans = [
            MinMaxResizeForTest(crop_size, param['test_respect_ratio_max'])
        ]
    else:
        trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),

        ]
    trans.extend([
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    transforms = Compose(trans)
    return transforms

def multi_video_inference(videos_csv, annotations_json_path, model_path, model_name, prefixes=None):
    
    """
    annotations_json_path: path to json containing original video annotataions
    """
    
    video_files_df = pd.read_csv(videos_csv)
    video_files = list(video_files_df['image_files'])
    video_files = [literal_eval(i) for i in video_files]
    
    
    if prefixes is None:
        prefixes = ['']*len(video_files)
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # model
    model = get_git_model(tokenizer, param)
    # pretrained = f'output/{model_name}/snapshot/model.pt'
    pretrained = model_path

    epoch_number = re.search(r'(epoch\d+)',model_path)
    if epoch_number is None: #inference from base model
        checkpoint = torch_load(pretrained)['model']
        load_state_dict(model, checkpoint)        
    else: #inference from our (MLP) model
        epoch_number = epoch_number.group()
        model.load_state_dict(torch.load(pretrained))

    predictions_file = "predictions_{}.json".format(str(epoch_number))
    metrics_file = "metrics_{}.csv".format(str(epoch_number))
    
    
    model.cuda()
    model.eval()

    vid_to_caption = {"videos": [], "sentences": []}
    for video_file, prefix in zip(video_files, prefixes):
        img = [load_image_by_pil(i) for i in video_file]

        transforms = get_image_transform(param)
        img = [transforms(i) for i in img]

        
        img = [i.unsqueeze(0).cuda() for i in img]

        # prefix
        max_text_len = 40
        prefix_encoding = tokenizer(prefix,
                                    padding='do_not_pad',
                                    truncation=True,
                                    add_special_tokens=False,
                                    max_length=max_text_len)
        payload = prefix_encoding['input_ids']
        if len(payload) > max_text_len - 2:
            payload = payload[-(max_text_len - 2):]
        input_ids = [tokenizer.cls_token_id] + payload

        with torch.no_grad():
            result = model({
                'image': img,
                'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
            })
        cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
        logging.info('output: {}'.format(cap))
        
        video_file_name = op.split(video_file[0])[-1].split('_')[0]
        
        vid_to_caption["videos"].append({'video_id': video_file_name})
        vid_to_caption["sentences"].append({'video_id': video_file_name,
                                            'caption': cap})
    
    # write dictionary to json
    
    with open(os.path.join(os.getcwd(),predictions_file), "w") as f:
        json.dump(vid_to_caption, f)

    # evaluate metrics
    metrics_obj = EvalCap(os.path.join(os.getcwd(),"predictions.json"),
                          annotations_json_path)
    metrics_obj.evaluate()
    
    # save metrics
    metrics = pd.DataFrame(data={'Metric Name':[], 'Metric Value':[]})
    for metric, score in metrics_obj.eval.items():
        metrics.append({'Metric Name':metric, 'Metric Value':score}, ignore_index=True)
    
    metrics.to_csv(metrics_file, index=False)

def multi_video_inference_dir(videos_csv, annotations_json_path, model_dir, model_name, prefixes=None):
    
    """
    annotations_json_path: path to json containing original video annotataions
    """
    #create directory fore prediction and metrics in model_dir
    os.mkdir(os.path.join(model_dir, 'predictions'))                 
    
    video_files_df = pd.read_csv(videos_csv)
    video_files = list(video_files_df['image_files'])
    video_files = [literal_eval(i) for i in video_files]
    
    for model_path in os.listdir(model_dir):
        
        if prefixes is None:
            prefixes = ['']*len(video_files)
        param = {}
        if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
            param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        # model
        model = get_git_model(tokenizer, param)
        # pretrained = f'output/{model_name}/snapshot/model.pt'
        pretrained = model_path
        epoch_number = re.search(r'(epoch\d+)',model_path)
        if epoch_number is None: #inference from base model
            checkpoint = torch_load(pretrained)['model']
            load_state_dict(model, checkpoint)        
        else: #inference from our (MLP) model
            epoch_number = epoch_number.group()
            model.load_state_dict(torch.load(pretrained))

        predictions_file = os.path.join(model_dir, 'predictions', "predictions_{}.json".format(str(epoch_number)))
        metrics_file = os.path.join(model_dir, 'predictions', "metrics_{}.csv".format(str(epoch_number)))

        # checkpoint = torch_load(pretrained)['model']
        # load_state_dict(model, checkpoint)
        # model.load_state_dict(torch.load(pretrained))
        model.cuda()
        model.eval()

        vid_to_caption = {"videos": [], "sentences": []}
        for video_file, prefix in zip(video_files, prefixes):
            img = [load_image_by_pil(i) for i in video_file]

            transforms = get_image_transform(param)
            img = [transforms(i) for i in img]

            
            img = [i.unsqueeze(0).cuda() for i in img]

            # prefix
            max_text_len = 40
            prefix_encoding = tokenizer(prefix,
                                        padding='do_not_pad',
                                        truncation=True,
                                        add_special_tokens=False,
                                        max_length=max_text_len)
            payload = prefix_encoding['input_ids']
            if len(payload) > max_text_len - 2:
                payload = payload[-(max_text_len - 2):]
            input_ids = [tokenizer.cls_token_id] + payload

            with torch.no_grad():
                result = model({
                    'image': img,
                    'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
                })
            cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
            logging.info('output: {}'.format(cap))
            
            video_file_name = op.split(video_file[0])[-1].split('_')[0]
            
            vid_to_caption["videos"].append({'video_id': video_file_name})
            vid_to_caption["sentences"].append({'video_id': video_file_name,
                                                'caption': cap})
        
        # write dictionary to json
        
        with open(os.path.join(os.getcwd(),predictions_file), "w") as f:
            json.dump(vid_to_caption, f)

        # evaluate metrics
        metrics_obj = EvalCap(os.path.join(os.getcwd(),"predictions.json"),
                            annotations_json_path)
        metrics_obj.evaluate()
        
        # save metrics
        metrics = pd.DataFrame(data={'Metric Name':[], 'Metric Value':[]})
        for metric, score in metrics_obj.eval.items():
            metrics.append({'Metric Name':metric, 'Metric Value':score}, ignore_index=True)
        
        metrics.to_csv(metrics_file, index=False)

if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

