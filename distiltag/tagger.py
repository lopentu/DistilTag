import os
import re
import torch
import numpy as np
from .download import get_model_path
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification
from collections import Counter

HALF2FULL = dict((i, i + 0xFEE0) for i in range(0x21, 0x7F))
HALF2FULL[0x20] = 0x3000

class DistilTag:
    def __init__(self, model_path=None, use_cuda=False):
        if not model_path:
            model_path = get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError("Cannot find model under " + model_path)
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.label_map = self.get_label_map(model_path)
        self.pos_list = sorted(list(set(x[2:] for x in self.label_map.values())))

    def get_label_map(self, model_path):
        label_path = os.path.join(model_path, "labels.txt")
        with open(label_path, "r") as f:
            labels = f.read().splitlines()
            label_map = {i: label for i, label in enumerate(labels)}
        return label_map

    def make_sentences(self, text):        
        sentence_pat = re.compile("([！：，。？；）：])")
        parts = sentence_pat.split(text.strip())
        try:
            sentence_pat = re.compile("([！：，。？；）：])")            
            sentences = []
            for i in range(0, len(parts), 2):                
                if not parts[i]: continue
                sentence = parts[i]
                if i+1 < len(parts):
                    sentence += parts[i+1]
                sentence = sentence.replace(" ", "_")                
                sentences.append(sentence)
        except:
            sentences = [text]
        return sentences

    def make_tagged(self, text, pred):
        tagged = []
        buf = ["", ""]        
        for ch, label in zip(text, pred):
            if label.startswith("B-") and buf[0]:
                word = buf[0]
                tagged.append(tuple(buf))
                buf = ["", ""]
            buf[0] += ch.replace("#", "").replace("_", " ").strip()
            buf[1] = label[2:]
        if buf[0]:
            tagged.append(tuple(buf))
        return tagged

    def tag(self, text, return_logits=False):
        text = text.translate(HALF2FULL)
        label_map = self.label_map
        sentences = self.make_sentences(text)
        tokenizer = self.tokenizer
        batch = tokenizer(sentences,    
                    return_offsets_mapping=True,                
                    return_tensors="pt",
                    padding=True)

        if self.device == "cuda":
            for k, v in batch.items():
                if k == "offset_mapping":
                    continue
                batch[k] = v.to(self.device)

        with torch.no_grad():
            outs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])[0]
            logits = outs
            preds = np.argmax(outs.cpu().numpy(), axis=2)

        batch_size, seq_len = preds.shape
        input_ids = batch["input_ids"].cpu().numpy()
        preds_list = [[] for _ in range(batch_size)]
        tokens_list = [[] for _ in range(batch_size)]        
        if return_logits:
            logits_list = [[] for _ in range(batch_size)]
        offset_mapping = batch["offset_mapping"]
        for i in range(batch_size):
            for j in range(1,(seq_len-1)):                
                if input_ids[i, j] in (101, 102, 0):
                    continue
                omap = offset_mapping[i][j]
                preds_list[i].append(label_map[preds[i][j]])
                if return_logits:
                    logits_list[i].append(logits[i, j, :])
                token = sentences[i][omap[0]:omap[1]]
                tokens_list[i].append(token)
        
        tagged_list = [self.make_tagged(tokens, pred)
                        for tokens, pred
                        in zip(tokens_list, preds_list)]
        
        if return_logits:
            logits_list = [torch.stack(tensor_xs) for tensor_xs in logits_list]
            return tagged_list, logits_list, tokens_list
        else:
            return tagged_list

    def print_tagged_list(self, tagged_list):
        for tagged_x in tagged_list:
            print("\u3000".join(f"{x[0]}({x[1]})" for x in tagged_x))
    
    def soft_tag(self, text, Tw=2, Tp=2):
        out = self.tag(text, return_logits=True)
        logits = out[1]
        tokens = out[2]
        n_seq = len(logits)
        
        word_logits = [[] for _ in range(n_seq)]
        pos_logits = [[] for _ in range(n_seq)]
        for seq_i in range(n_seq):
            seq_logits = logits[0]
            M = seq_logits.shape[0]
            wb_table = np.zeros(shape=(M, 2), dtype=np.float32)
            pos_table = np.zeros(shape=(M, len(self.pos_list)), dtype=np.float32)

            label_map = self.label_map
            # word boundary
            wb = [idx for idx in label_map.keys() if label_map[idx].startswith("B-")]
            wi = [idx for idx in label_map.keys() if label_map[idx].startswith("I-")]
            wb_table[:, 0] = seq_logits[:, wb].mean(axis=1)
            wb_table[:, 1] = seq_logits[:, wi].mean(axis=1)
            wb_table = torch.from_numpy(wb_table)

            # pos
            for pos_i, pos_x in enumerate(self.pos_list):
                mask_idx = [idx for idx in label_map.keys() if label_map[idx].endswith(pos_x)]
                pos_table[:, pos_i] = seq_logits[:, mask_idx].mean(axis=1)
            pos_table = torch.from_numpy(pos_table)
            
            word_logits[seq_i] = wb_table
            pos_logits[seq_i] = pos_table

        return word_logits, pos_logits, tokens

    def logits_to_probs(self, logits, T=1):
        probs = [torch.softmax(logit_x/T, axis=1)
                 for logit_x in logits]
        return probs

    def decode_soft_tag(self, word_logits, pos_logits, tokens, 
            Tw=2, Tp=2, wcut=0.5):
        word_probs = self.logits_to_probs(word_logits, Tw)
        pos_probs = self.logits_to_probs(pos_logits, Tp)
        
        seq_words = []
        for word_prob, pos_prob, tokens_seq in zip(word_probs, pos_probs, tokens):
            word_list = []
            buf = []
            for tok_i in range(len(tokens_seq)):
                p_wb = word_prob[tok_i, 0]
                if p_wb > wcut and buf:
                    word = "".join([tokens_seq[i] for i in buf])
                    pos_idx = pos_prob[buf, :].mean(axis=0).argmax()
                    pos = self.pos_list[pos_idx]
                    word_list.append((word, pos))
                    buf = []
                buf.append(tok_i)
            word = "".join([tokens_seq[i] for i in buf])
            pos_idx = pos_prob[buf, :].mean(axis=0).argmax()
            pos = self.pos_list[pos_idx]
            word_list.append((word, pos))

            seq_words.append(word_list)
        return seq_words
    
    def print_soft_tag(self, word_logits, pos_logits, tokens, Tw=2, Tp=2):
        word_probs = self.logits_to_probs(word_logits, Tw)
        pos_probs = self.logits_to_probs(pos_logits, Tp)

        def show_pos(i):
            return self.pos_list[i].replace("CATEGORY", "")[:3]
        for word_prob, pos_prob, tokens_seq in zip(word_probs, pos_probs, tokens):
            for tok_i, tok_x in enumerate(tokens_seq):    
                probs, inds = pos_prob[tok_i, :].topk(5)
                wb = word_prob[tok_i, 0]
                pos_strips = [f"{show_pos(i):>3s}_{prob:.2f}" 
                                for i, prob in zip(inds, probs)]
                out_str = tok_x + f"_{wb:.2f} " + "/".join(pos_strips)
                print(out_str)
            print("")