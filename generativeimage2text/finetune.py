from .common import Config
import json
import re
from math import ceil
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
from .vc_inference import multi_video_inference

def get_data(video_file, prefix, target, tokenizer, param):
    max_text_len = 40

    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    
    target_encoding = tokenizer(
        target, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    
    need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
    payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
    
    if len(payload) > max_text_len:
        payload = payload[-(max_text_len - 2):]
        need_predict = need_predict[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]

    img = [load_image_by_pil(i) for i in video_file]

    transforms = get_image_transform(param)
    img = [transforms(i) for i in img]
    # img = [i.unsqueeze(0).cuda() for i in img]


    data = {
        'caption_tokens': torch.tensor(input_ids),
        #'caption_lengths': len(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': img,
        # 'rect' field can be fed in 'caption', which tells the bounding box
        # region of the image that is described by the caption. In this case,
        # we can optionally crop the region.
        'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        'iteration': 0,
    }
    

    return data

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


def forward_backward(video_files, model_name, captions, prefixes=None):
    if prefixes is None:
        prefixes = [''] * len(captions)
    param = {}
    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
   
    # model
    model = get_git_model(tokenizer, param)
    pretrained = 'model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    print('base model setup succesful!')

    # max_text_len = 40
    all_data = []
    for video_file, prefix, target in zip(video_files, prefixes, captions):
        #print(video_file)
        data = get_data(video_file, prefix, target, tokenizer, param)
        all_data.append(data)
    print('batch data transformed successfully')
    data = collate_fn(all_data)
    data = recursive_to_device(data, 'cuda')
    print('batch data collated and moved to cuda')
    
    model.train()
    model.cuda()
    print('model moved to cuda')
    loss_dict = model(data)
    loss = sum(loss_dict.values())
    loss.backward()
    logging.info(loss)
    return loss
    # img = [i.unsqueeze(0).cuda() for i in img]

def get_batches(full_list, batch_size):
    batches = []
    for i in range(ceil(len(full_list)/batch_size)):
        batches.append(full_list[i*batch_size : (i+1)*batch_size])
    return batches

def write_loss(loss, loss_type):
    if loss_type == 'validation':
        filename = 'val_losses.csv'
    elif loss_type == 'train':
        filename = 'train_losses.csv'

    with open(filename, 'a') as f:
        f.write(str(loss) + '\n')

def get_val_loss(validation_csv, model, tokenizer, param):
    model.eval()
    model.validation = True #model attribute to calculate loss
    vid_caption_df = pd.read_csv(validation_csv)
    video_files = list(vid_caption_df['image_files'])
    video_files = [literal_eval(i) for i in video_files]
    # print(video_files[0:4])
    captions = list(vid_caption_df['caption'])
    # print(len(video_files))
    prefixes = [''] * len(captions)

    #break data into batches
    video_file_batches = get_batches(video_files, batch_size=4)
    caption_batches = get_batches(captions, batch_size=4)
    prefix_batches = get_batches(prefixes, batch_size=4)
    print('validation data successfully batched')


    running_loss = 0.0
    i = 0 #batch number tracker        
    #minibatch prediction on validation data
    for video_files_batch, prefix_batch, captions_batch in zip(video_file_batches, prefix_batches, caption_batches):

        #transform minibatch data
        batch_data = []
        for video_file, prefix, target in zip(video_files_batch, prefix_batch, captions_batch):
            #print(video_file)
            data = get_data(video_file, prefix, target, tokenizer, param)
            batch_data.append(data)
        print('batch data {} transformed successfully'.format(str(i)))

        data = collate_fn(batch_data)
        data = recursive_to_device(data, 'cuda')
        print('batch data collated and moved to cuda')

        #obtain loss for batch
        with torch.no_grad():
          loss_dict = model(data)
        #   del loss_dict['predictions']
        #   del loss_dict['logprobs']
          loss = sum(loss_dict.values())
          #print(loss.item())
          running_loss += loss
        #   print ('running loss {}'.format(str(running_loss.item() )))
        i += 1
    print('validation data completly evaluated')
    avg_loss = running_loss/len(video_files)
    return avg_loss


def train(train_csv, validation_csv, validation_annotations_json, model_name, model_path, batch_size, epochs, threshold=0.0001, prefixes=None):

    vid_caption_df = pd.read_csv(train_csv)
    video_files = list(vid_caption_df['image_files'])
    video_files = [literal_eval(i) for i in video_files]
    # print(video_files[0:4])
    captions = list(vid_caption_df['caption'])
    # print(len(video_files))
    if prefixes is None:
        prefixes = [''] * len(captions)
    
    param = {}

    if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
   
    # model
    model = get_git_model(tokenizer, param)
    pretrained = model_path
    epoch_number = re.search(r'(\d+)',model_path)
    if epoch_number is None: #starting training from base model
        epoch_number = -1
        checkpoint = torch_load(pretrained)['model']
        load_state_dict(model, checkpoint)
    else: #starting from our(MLP) saved model
        epoch_number = int(epoch_number.group())
        model.load_state_dict(torch.load(pretrained))   
    
    
    print('base model setup succesful!')
    model.cuda()
    print('model moved to cuda')

    #use adam optimiser
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.00001)
    train_losses = [] #keep track of validation losses through all epochs
    val_losses = [] #keep track of validation losses through all epochs
    
    best_val_loss = 1000000.

    for epoch in range(epochs):
        if epoch <= epoch_number:
            continue
        
        print('epoch ', str(epoch+1))
        model.train(True)
        #shuffle training data for current epoch
        train_permutations = torch.randperm(len(video_files))
        # print(train_permutations[:10])
        shuffled_video_files = [video_files[p] for p in train_permutations]
        shuffled_captions = [captions[p] for p in train_permutations]
        shuffled_prefixes = [prefixes[p] for p in train_permutations]

        #break data into batches
        video_file_batches = get_batches(shuffled_video_files, batch_size)
        caption_batches = get_batches(shuffled_captions, batch_size)
        prefix_batches = get_batches(shuffled_prefixes, batch_size)
        print('data successfully batched')
        
        batch_losses = []   #keep track of losses through all batches
        running_loss = 0.0
        i = 0 #batch number tracker        
        #minibatch training on training_set
        for video_files_batch, prefix_batch, captions_batch in zip(video_file_batches, prefix_batches, caption_batches):
            # print(len(video_files_batch))
            print('epoch {} batch {}/{}'.format(str(epoch), str(i), len(video_file_batches)))

            #transform minibatch data
            batch_data = []
            for video_file, prefix, target in zip(video_files_batch, prefix_batch, captions_batch):
                #print(video_file)
                data = get_data(video_file, prefix, target, tokenizer, param)
                batch_data.append(data)
            print('batch data transformed successfully')

            data = collate_fn(batch_data)
            data = recursive_to_device(data, 'cuda')
            print('batch data collated and moved to cuda')
            
            #train model
            optimizer.zero_grad()
            loss_dict = model(data)
            loss = sum(loss_dict.values())
            # print(loss_dict.keys())
            running_loss += loss 
            loss.backward()
            optimizer.step()
            logging.info(loss)
            batch_losses.append(loss)
            i += 1
        
        avg_train_loss = running_loss / len(video_files)
        train_losses.append(avg_train_loss)
        write_loss(avg_train_loss.item(), 'train')
        # running_loss = 0.0
        
        ## evaluation on validation dataset
        #avg_val_loss = get_val_loss(validation_csv, model, tokenizer, param)
        #val_losses.append(avg_val_loss)
        #write_loss(avg_val_loss.item(), 'validation')

        
        #print('epoch {}, avg_train_loss: {}, avg_val_loss: {}'.format(str(epoch+1), avg_train_loss, avg_val_loss))
        #if torch.abs(avg_val_loss - best_val_loss) < threshold:
            ##saved model name for this epoch
            #saved_model_name = 'best_msrvtt_model_epoch{}.pt'.format(str(epoch))
            ##save model parameters
            #torch.save(model.state_dict(), saved_model_name)
            #break

        #if avg_val_loss < best_val_loss:
            #best_val_loss = avg_val_loss
            ##saved model name for this epoch
            #saved_model_name = 'better_msrvtt_model_epoch{}.pt'.format(str(epoch))
            ##save model parameters
            #torch.save(model.state_dict(), saved_model_name)  
        #for epochs that did not improve in terms of validation loss  
        #else:            
            ##saved model name for this epoch
            #saved_model_name = 'regular_msrvtt_model_epoch{}.pt'.format(str(epoch))
            ##save model parameters
            #torch.save(model.state_dict(), saved_model_name) 
        # 
        # saved model epoch
        saved_model_name = 'regular_msrvtt_model_epoch{}.pt'.format(str(epoch))
        # save model parameters
        torch.save(model.state_dict(), saved_model_name)

        #save evaluation metrics on validation set
        # multi_video_inference(validation_csv, validation_annotations_json, saved_model_name, model_name)  
        
    print('Average training losses over all epochs: {}'.format(str(train_losses)))    
    #print('Average validation losses over all epochs: {}'.format(str(val_losses)))   
    
    
             

   
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

def test_git_inference_single_tsv(image_tsv, model_name, question_tsv, out_tsv):
    param = {}
    if File.isfile(f'output/{model_name}/parameter.yaml'):
        param = load_from_yaml_file(f'output/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_tsv = TSVFile(image_tsv)
    question_tsv = TSVFile(question_tsv) if question_tsv else None

    transforms = get_image_transform(param)

    # model
    model = get_git_model(tokenizer, param)
    pretrained = f'output/{model_name}/snapshot/model.pt'
    checkpoint = torch_load(pretrained)['model']
    load_state_dict(model, checkpoint)
    model.eval()

    torch.cuda.set_device(get_mpi_local_rank())
    model.cuda()

    # prefix
    max_text_len = 40
    rank = get_mpi_rank()
    world_size = get_mpi_size()
    def get_rank_specific_tsv(rank):
        return '{}.{}.{}.tsv'.format(out_tsv, rank, world_size)
    if world_size > 1:
        curr_out_tsv = get_rank_specific_tsv(rank)
    else:
        curr_out_tsv = out_tsv
    total = len(image_tsv)
    curr_size = (total + world_size - 1) // world_size
    curr_start = curr_size * rank
    curr_end = curr_start + curr_size
    curr_end = min(curr_end, total)

    if question_tsv:
        def gen_rows():
            for i  in tqdm(range(curr_start, curr_end)):
                image_key, image_col = image_tsv[i]
                q_key, q_info = question_tsv[i]
                assert image_key == q_key
                q_info = json.loads(q_info)
                img = pilimg_from_base64(image_col)
                img = transforms(img)
                img = img.cuda().unsqueeze(0)
                for q in q_info:
                    prefix = q['question']
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
                    answer = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
                    result = {'answer': answer, 'question_id': q['question_id']}
                    yield json_dump(result),
    else:
        def gen_rows():
            for i  in tqdm(range(curr_start, curr_end)):
                key, col = image_tsv[i]
                img = pilimg_from_base64(col)
                img = transforms(img)
                img = img.cuda().unsqueeze(0)
                with torch.no_grad():
                    result = model({
                        'image': img,
                    })
                cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
                yield key, json_dump([{'caption': cap}])
    tsv_writer(gen_rows(), curr_out_tsv)
    if world_size > 1 and rank == 0:
        all_sub_tsv = [get_rank_specific_tsv(i) for i in range(world_size)]
        while True:
            not_ready = [t for t in all_sub_tsv if not File.isfile(t)]
            if len(not_ready) == 0:
                break
            else:
                import time
                logging.info('waiting {}'.format(','.join(not_ready)))
                time.sleep(5)
        from .tsv_io import concat_tsv_files
        concat_tsv_files(all_sub_tsv, out_tsv)

def convert_tsv_to_vqa_json(predict_file, out_json):
    result = [json.loads(s) for s, in tsv_reader(predict_file)]
    write_to_file(json_dump(result), out_json)

def convert_tsv_to_coco_format(res_tsv, outfile,
        sep='\t', key_col=0, cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                if len(caps) == 0:
                    caps = [{'caption': ''}]
                assert len(caps) == 1, 'cannot evaluate multiple captions per image'
                cap = caps[0]['caption']
            else:
                # empty caption generated
                cap = ""
            results.append(
                    {'image_id': key,
                    'caption': cap}
                    )
    with open(outfile, 'w') as fp:
        json.dump(results, fp)

def iter_caption_to_json(iter_caption, json_file):
    # save gt caption to json format so thet we can call the api
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        'info': 'dummy',
        'licenses': 'dummy',
        'type': 'captions',
    }
    info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({
                'image_id': k,
                'caption': c['caption'],
                'id': n
            })
            n += 1
    info['annotations'] = annotations
    write_to_file(json.dumps(info), json_file)

def evaluate_on_coco_caption(res_file, label_file, outfile=None,
                             ):
    if not outfile:
        outfile = op.splitext(res_file)[0] + '.eval.json'

    if res_file.endswith('.tsv'):
        res_file_coco = op.splitext(res_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(res_file, res_file_coco)
    else:
        res_file_coco = res_file

    if label_file.endswith('.tsv'):
        json_caption = '/tmp/{}'.format(label_file)
        iter_caption_to_json(
            TSVFile(label_file),
            json_caption)
        label_file = json_caption

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes)

    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result

if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)



