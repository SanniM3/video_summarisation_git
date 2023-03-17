import json
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
# from meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
# from spice.spice import Spice

class EvalCap:
    def __init__(self, captions_json, results_json):
        self.captions_dict = self.make_captions_dict(captions_json)
        self.results_dict = self.make_captions_dict(results_json)
        
        for k in list(self.captions_dict.keys()):
            if k not in self.results_dict:
                del self.captions_dict[k]

        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
    
    def make_captions_dict(self, json_file, results=False):
        """ Assumes json annotation file as input which is of format:
            
        {
          "videos": [video],
          "sentences": [sentence],
        }
        
        video{
          "video_id": str,
          "category": int,
          "start time": float,
          "end time": float,
        }
        
        sentence{
          "video_id": str,
          "caption": str,
         }
        
        NB if results=True, only 'video_id' and 'caption' fields required.
        
        Returns dict of format:
            
        {
            video_id1: [sentence1, sentence2 ...],
            video_id2: [sentence1, sentence2 ...],
            ...
        }
        
        NB if results=True, list of captions will be of length 1.
        """
        
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        
        video_ids_with_anns = {item["video_id"]: [] for item in json_data["videos"]}
        for caption in json_data['sentences']:
            video_ids_with_anns[caption['video_id']] += [caption]
        
        return video_ids_with_anns

    def evaluate(self):
        gts = self.captions_dict
        res = self.results_dict

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]
