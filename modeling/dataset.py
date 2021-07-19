import json
from pathlib import Path
import torch
import clip
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import os
import pickle
from dataclasses import dataclass
import random


def preprocess_data(path, img_path, device, tokenizer, train_flag=False, train_size=1, text_only=False, use_clip=False):
    """
    Reads data from a json file and prepares datasets for sequence classification.
    Returns large and small-sized datasets.
    """
    with open(Path(path), 'rb') as f:
        data = json.load(f)

    prompts = []
    answers = []
    labels = []
    img_feat = []
    img_pos_feat = []
    img_person_ids = []
    clip_model, preprocess = clip.load('ViT-B/32', device)

    if train_flag and train_size >= 2:
        data = data[:int(len(data) // train_size)]

    img_id = ''
    image_features = None
    boxes = None
    person_ids = None
    for i, example in enumerate(data):
        ex_img_id = example['img_id']

        if not text_only:
            if use_clip:
                if ex_img_id != img_id:
                    img_id = ex_img_id
                    bbox_img = Image.open('../data/bboxes/{}'.format(ex_img_id))
                    bbox_image_input = preprocess(bbox_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(bbox_image_input)  # torch.Size([1, 512])
                img_feat.append(image_features)
            else:  # using object features
                if ex_img_id != img_id:
                    img_id = ex_img_id

                    pos_id = img_id[:img_id.rfind('.')]
                    with open(os.path.join(img_path['img_dir'], pos_id) + '.json', 'r') as f:
                        metadata = json.load(f)

                    # consider all objects that are people
                    objs = metadata['names']
                    objs.append('')
                    last_person = objs.index(next(filter(lambda k: k != 'person', objs)))

                    # Chop off the final dimension, that's the confidence
                    boxes = torch.tensor(metadata['boxes'])[:last_person, :-1]
                    w = (boxes[:, 2] - boxes[:, 0]).unsqueeze(1)
                    h = (boxes[:, 3] - boxes[:, 1]).unsqueeze(1)
                    boxes = torch.cat([boxes, w, h, w * h], dim=1)
                    person_ids = ['<|det%d|>' % p_id for p_id in range(1, last_person + 1)]
                    person_ids = tokenizer(person_ids)
                    person_ids = torch.tensor(person_ids['input_ids'])

                    ft_id = img_id[img_id.rfind('/') + 1:img_id.rfind('.')]
                    with open(os.path.join(img_path['ft_dir'], ft_id) + '.pkl', 'rb') as p:
                        features_dict = pickle.load(p)

                    image_features = features_dict['object_features'][:last_person]  # (num_bboxes, 2048)

                img_feat.append(torch.tensor(image_features))
                img_pos_feat.append(boxes)
                img_person_ids.append(person_ids)

        # prompts.append([example['event'].split('.')[-1]] * 4)  # don't include event in the mc prompt
        # answers.append([example['ending0'], example['ending1'], example['ending2'], example['ending3']])
        prompts.append(['a'] * 4)
        answers.append([str(random.randint(0, 1000000)), str(random.randint(0, 1000000)), str(random.randint(0, 1000000)), str(random.randint(0, 1000000))])
        labels.append(example['label'])

        if i % 1000 == 0:
            print(path, 'Example:', i)
    return prompts, answers, labels, img_feat, img_pos_feat, img_person_ids


class TextMCDataset(Dataset):
    """
    Text-only multiple choice dataset
    """
    def __init__(self, prompts, answers, labels):
        self.prompts = prompts
        self.answers = answers
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item['prompt'] = self.prompts[idx]  # [p, p, p, p]
        item['answer'] = self.answers[idx]  # [a1, a2, a3, a4]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class DataCollatorForTextMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for text-only multiple choice received.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompt_batch = []
        answer_batch = []
        labels_batch = []
        batch_size = len(batch)
        num_choices = len(batch[0]['prompt'])

        for example in batch:
            prompt_batch.extend(example['prompt'])
            answer_batch.extend(example['answer'])
            labels_batch.append(example['labels'])
        encodings = self.tokenizer(prompt_batch, answer_batch, padding=True, truncation=True, return_tensors="pt")
        new_batch = {k: v.view(batch_size, num_choices, -1) for k, v in encodings.items()}
        new_batch['labels'] = torch.tensor(labels_batch)

        return new_batch


class VL_MC_Dataset(Dataset):
    """
    Vision and Language multiple choice dataset
    """
    def __init__(self, prompts, answers, labels, img_feat, img_pos_feat, person_ids):
        self.prompts = prompts
        self.answers = answers
        self.labels = labels
        self.img_feat = img_feat
        self.img_pos_feat = img_pos_feat
        self.person_ids = person_ids

    def __getitem__(self, idx):
        item = {}
        item['prompt'] = self.prompts[idx]  # [p, p, p, p]
        item['answer'] = self.answers[idx]  # [a1, a2, a3, a4]
        item['labels'] = torch.tensor(self.labels[idx]).float().unsqueeze(0)
        item['img_feat'] = self.img_feat[idx]
        # #item['img_feat'] = torch.tensor([False]) if self.img_feat is None else self.img_feat[idx]  # torch.tensor(self.img_feat[idx])
        item['img_pos_feat'] = self.img_pos_feat[idx].clone()
        item['person_ids'] = self.person_ids[idx].clone()
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class DataCollatorForVLMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for vision and language multiple choice received.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompt_batch = []
        answer_batch = []
        img_feat_batch = []
        img_pos_feat_batch = []
        person_ids_batch = []
        labels_batch = []
        batch_size = len(batch)
        for example in batch:
            prompt_batch.extend(example['prompt'])
            answer_batch.extend(example['answer'])
            img_feat_batch.append(example['img_feat'])
            img_pos_feat_batch.append(example['img_pos_feat'])
            person_ids_batch.append(example['person_ids'])
            labels_batch.append(example['labels'])
        num_choices = 4

        encodings = self.tokenizer(prompt_batch, answer_batch, padding=True, truncation=True, return_tensors="pt")

        input_ids_batch = encodings['input_ids']
        position_ids = torch.arange(0, input_ids_batch.size(1), dtype=torch.long).unsqueeze(0).long()

        num_bbs = [f.shape[0] for f in img_feat_batch]
        img_attn_mask = torch.repeat_interleave(
            pad_sequence([torch.ones(i) for i in num_bbs], batch_first=True), 4, dim=0
        )
        attn_mask_batch = torch.cat([encodings['attention_mask'], img_attn_mask], dim=1)

        img_feat_batch = torch.repeat_interleave(pad_tensors(img_feat_batch, num_bbs), 4, dim=0)
        img_pos_feat_batch = torch.repeat_interleave(pad_tensors(img_pos_feat_batch, num_bbs), 4, dim=0)
        person_ids_batch = torch.repeat_interleave(pad_tensors(person_ids_batch, num_bbs), 4, dim=0)

        batch = {'input_ids': input_ids_batch.view(batch_size, num_choices, -1),
                 'position_ids': position_ids,
                 'img_feat': img_feat_batch,
                 'img_pos_feat': img_pos_feat_batch,
                 'person_ids': person_ids_batch,
                 'attention_mask': attn_mask_batch.view(batch_size, num_choices, -1),
                 'labels': torch.tensor(labels_batch)}
        return batch


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""

    if lens is None:
        lens = [t.shape[0] for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output



