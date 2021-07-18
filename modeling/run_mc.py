import argparse
import torch
import copy
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, AdamW
from tqdm import tqdm

from dataset import (
    preprocess_data, TextMCDataset, DataCollatorForTextMultipleChoice,
    VL_MC_Dataset, DataCollatorForVLMultipleChoice
)
from img_roberta_model import ImageRobertaForMultipleChoice

from logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter


parser = argparse.ArgumentParser(description='Text-Only Examples')

parser.add_argument('--batch-size', type=int, default=16, metavar='N')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--train-size', type=int, default=2)
parser.add_argument('--use-clip', type=bool, default=False)
parser.add_argument('--text-only', type=bool, default=False)
parser.add_argument('--load-from-checkpoint', type=bool, default=False)
parser.add_argument('--train_data_path', type=str, default='../data/train.json')
parser.add_argument('--val_data_path', type=str, default='../data/val.json')
parser.add_argument('--vcr-img-dir', type=str, default='../../visualcomet/vcr1images/')
parser.add_argument('--vcr-ft-dir', type=str, default='../../visualcomet/features/')
# parser.add_argument('--bbox-img-dir', type=str)
parser.add_argument('--best-model', type=str, default='test_roberta.pth')
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = RobertaTokenizerFast.from_pretrained(args.model)

vcr_data = {'img_dir': args.vcr_img_dir, 'ft_dir': args.vcr_ft_dir}

prompts_train, answers_train, labels_train, img_feat_train, img_pos_feat_train, img_person_ids_train = preprocess_data(
    args.train_data_path, vcr_data, device, tokenizer, train_flag=True, train_size=args.train_size
)  # 2348126 examples
prompts_val, answers_val, labels_val, img_feat_val, img_pos_feat_val, img_person_ids_val = preprocess_data(
    args.val_data_path, vcr_data, device, tokenizer
)  # 292664 examples


if args.use_clip:
    large_train_dataset = VL_MC_Dataset(
        prompts_train, answers_train, labels_train, img_feat_train, img_pos_feat_train, img_person_ids_train
    )
    large_val_dataset = VL_MC_Dataset(
        prompts_val, answers_val, labels_val, img_feat_val, img_pos_feat_val, img_person_ids_val
    )
elif args.text_only:
    large_train_dataset = TextMCDataset(prompts_train, answers_train, labels_train)
    large_val_dataset = TextMCDataset(prompts_val, answers_val, labels_val)
else:  # object features
    large_train_dataset = VL_MC_Dataset(
        prompts_train, answers_train, labels_train, img_feat_train, img_pos_feat_train, img_person_ids_train
    )
    large_val_dataset = VL_MC_Dataset(
        prompts_val, answers_val, labels_val, img_feat_val, img_pos_feat_val, img_person_ids_val
    )

LOGGER.info("device: {} num_gpus: {}, lr: {}, batch size: {}".format(
    device, torch.cuda.device_count(), args.lr, args.batch_size
))
print('Using', torch.cuda.device_count(), 'GPUs')
print('batch size:', args.batch_size)
print('lr:', args.lr)
print('model:', args.model)
print('training size:', args.train_size)

LOGGER.info("Initializing train_loader and val_loader...")
if args.text_only:
    large_train_loader = DataLoader(
        large_train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForTextMultipleChoice(tokenizer=RobertaTokenizerFast.from_pretrained(args.model))
    )
    large_val_loader = DataLoader(
        large_val_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForTextMultipleChoice(tokenizer=RobertaTokenizerFast.from_pretrained(args.model))
    )
else:
    large_train_loader = DataLoader(
        large_train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForVLMultipleChoice(tokenizer=RobertaTokenizerFast.from_pretrained(args.model))
    )
    large_val_loader = DataLoader(
        large_val_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
        collate_fn=DataCollatorForVLMultipleChoice(tokenizer=RobertaTokenizerFast.from_pretrained(args.model))
    )

optim = AdamW(model.parameters(), lr=args.lr)
num_epochs = 3

LOGGER.info("Setting up model...")
if args.load_from_checkpoint:
    model = ImageRobertaForMultipleChoice.from_pretrained("../config/uniter-base.json")
    checkpoint = torch.load('../models/{}'.format(args.best_model))
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epochs -= checkpoint['epoch']
if args.text_only:
    model = ImageRobertaForMultipleChoice.from_pretrained("../config/uniter-base.json")
else:
    model = ImageRobertaForMultipleChoice.from_pretrained(
        "../config/uniter-base.json", img_dim=2048 if not args.use_clip else 512, num_answer=1
    )

# model = nn.DataParallel(model)  # , device_ids=[0, 1]).cuda()
model.to(device)
LOGGER.info("Setup model done!")


def train(model, optimizer, train_loader, val_loader, num_epochs, best_model_path):
    best_accuracy_val = 0
    best_accuracy_train = 0

    for epoch in range(num_epochs):
        # Total loss across train data
        train_loss = 0
        # Total number of correctly predicted training labels
        train_correct = 0
        # Total number of training sequences processed
        train_seqs = 0

        tqdm_train_loader = tqdm(train_loader)
        print(f"Epoch {epoch + 1} / {num_epochs}")

        model.train()
        for batch_idx, batch in enumerate(tqdm_train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if args.text_only:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                img_feat = batch['img_feat'].to(device)
                if args.use_clip:
                    outputs = model(input_ids, img_feat, attention_mask, labels)
                else:
                    position_ids = batch['position_ids'].to(device)
                    img_pos_feat = batch['img_pos_feat'].to(device)
                    outputs = model(input_ids, img_feat, attention_mask, labels, position_ids, img_pos_feat)

            loss = outputs['loss']
            correct = outputs['correct'].detach()
            labels = labels.detach()
            loss.backward()
            optimizer.step()

            # Accumulate metrics and update status
            train_loss += loss.detach().item()
            train_correct += correct.sum()  # .item()
            train_seqs += len(labels)
            tqdm_train_loader.set_description_str(
                f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {(train_correct / train_seqs):.4f}")
        print()

        avg_train_loss = train_loss / len(tqdm_train_loader)
        train_accuracy = train_correct / train_seqs
        print(f"[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}")

        print("Validating")
        # Total loss across validation data
        val_loss = 0
        # Total number of correctly predicted validation labels
        val_correct = 0
        # Total number of validation sequences processed
        val_seqs = 0

        tqdm_val_loader = tqdm(val_loader)

        model.eval()
        for batch_idx, batch in enumerate(tqdm_val_loader):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                if args.text_only:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    img_feat = batch['img_feat'].to(device)
                    if args.use_clip:
                        outputs = model(input_ids, img_feat, attention_mask, labels)
                    else:
                        position_ids = batch['position_ids'].to(device)
                        img_pos_feat = batch['img_pos_feat'].to(device)
                        outputs = model(input_ids, img_feat, attention_mask, labels, position_ids, img_pos_feat)

                loss = outputs['loss'].mean()
                correct = outputs['correct'].detach()

                # Compute loss and number of correct predictions and accumulate metrics and update status
                val_loss += loss.detach().item()
                val_correct += correct
                val_seqs += len(labels.detach())
                tqdm_val_loader.set_description_str(
                    f"[Loss]: {val_loss / (batch_idx + 1):.4f} [Acc]: {(val_correct / val_seqs):.4f}")
        print()

        avg_val_loss = val_loss / len(tqdm_val_loader)
        val_accuracy = val_correct / val_seqs
        print(f"[Validation Loss]: {avg_val_loss:.4f} [Validation Accuracy]: {val_accuracy:.4f}")

        # find best training and validation accuracy,
        if train_accuracy > best_accuracy_train:
            best_accuracy_train = train_accuracy
        if val_accuracy > best_accuracy_val:
            best_accuracy_val = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)

    print('Finished Training')
    print(f"[Best Training Accuracy]: {best_accuracy_train:.4f}")
    print(f"[Best Validation Accuracy]: {best_accuracy_val:.4f}")


best_model_path = '../models/{}'.format(args.best_model)
train(model, optim, large_train_loader, large_val_loader, num_epochs, best_model_path)


import json
import random
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def _split_tokens(sentence):
    return sentence.replace(',', ' , ').replace("'", " '").replace('.', ' .').replace('?', ' ?').split()


def _get_event_tokens(sentence):
    """
    Creates the event statement by replacing person labels in the given sentence
    with special tokens. Returns the processed event statement, subject of the sentence,
    event ids, and subject ids.
    """
    tokens = _split_tokens(sentence)
    subject = [tokens[0]]   # initially assume subject is first token
    if tokens[0] == 'the':
        # subject is first two words of the sentence
        subject = tokens[:2]
    elif 'and' in tokens:
        # include all people in the subject
        and_idx = tokens.index('and')
        # check if all tokens before and_idx and the token after and_idx are digits.
        # Include all these tokens in the subject
        if tokens[and_idx + 1].isdigit():
            # check if subject is in "1, 2, ..., n, and n+1" format.
            # otherwise "and" is not part of the subject.
            if len([token for token in tokens[:and_idx] if token.isdigit()]) == len(tokens[:and_idx]):
                # "1, 2, ..., n, and n+1" is the subject
                subject = tokens[:and_idx + 2]

    subject_ids_set = set()
    for t in subject:
        if t.isdigit():
            subject_ids_set.add(t)

    event_ids_set = set()
    for t in tokens:
        if t.isdigit():
            event_ids_set.add(t)

    event_ids = list(event_ids_set)
    map_idx = {}
    # map person labels with special tokens
    for i in range(len(event_ids)):
        map_idx[event_ids[i]] = '<|det%d|>' % (int(event_ids[i]))

    tokens = [t if not t.isdigit() or t not in map_idx else map_idx[t] for t in tokens]
    subject = [t if not t.isdigit() or t not in map_idx else map_idx[t] for t in subject]
    new_event = ' '.join(tokens).strip().replace("'", " '").replace(' .', '.')
    subject = ' '.join(subject).strip().replace("'", " '").replace(' .', '.')
    return subject, new_event, subject_ids_set, event_ids_set


def _get_inference_tokens_and_ids(sentence, person_ids=None, subject_ids=None):
    """
    Creates an inference statement by replacing person labels in the given sentence
    with special tokens. Uses the given person_ids to replace person labels.
    Returns the processed inference statement and its subject.
    """
    tokens = _split_tokens(sentence)

    inference_ids_set = set()
    # if not given person_ids to replace person labels in sentence
    if person_ids is None:
        for t in tokens:
            if t.isdigit():
                inference_ids_set.add(t)

        person_ids = list(inference_ids_set)

        map_idx = {}
        # map person labels with special tokens
        for i in range(len(person_ids)):
            map_idx[person_ids[i]] = '<|det%d|>' % (int(person_ids[i]))
        tokens = [t if not t.isdigit() or t not in map_idx else map_idx[t] for t in tokens]
    elif len(person_ids) > 0:
        # create inference statement using the given person_ids
        new_tokens = []
        idx = 0
        for t in tokens:
            if t.isdigit():
                new_tokens.append('<|det%d|>' % (int(person_ids[idx])))
                idx += 1
            else:
                new_tokens.append(t)
        tokens = new_tokens

    processed_inference = ' '.join(tokens).strip().replace("'", " '").replace(' .', '.')

    # consider case where subject of event is not the subject of the inference.
    # we assume in general that the subject of the event would be the subject of the inference
    # and would not be the object of the inference.
    # but sometimes if the event has a compound subject and a subset of that subject is also
    # the object of the inference.
    # Ex:
    # "event": "1 and 2 eat Chinese takeout on the floor while looking through paperwork"
    # "after": ["talk with 2 about the paper"]
    # Here, the implied subject of the inference should be
    # subject_ids - set(person_ids) = {1, 2} - {2} = {1} = 1
    # instead of "1 and 2"
    subject_w_no_obj = None
    if subject_ids is not None:
        subject_w_no_obj = subject_ids.copy()
        if len(subject_ids - set(person_ids)) != 0:
            subject_w_no_obj = subject_ids - set(person_ids)
    return subject_w_no_obj, inference_ids_set, processed_inference


def read_text(path, large_file):
    """
    Reads data from a json file and prepares datasets for multiple choice.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        annots_dict = json.load(f)

    # get negative inference that is dissimilar to gt
    def get_negative_dissimilar(positive_inf, img_id, inference_type, person_ids):
        positive_emb = model.encode(positive_inf)

        while True:
            neg_idx = random.sample(list(range(len(annots_dict))), 1)[0]
            neg_example = annots_dict[neg_idx]

            # negative can't be from same image and must have inference type
            if neg_example['img_fn'] == img_id or inference_type not in neg_example:
                continue

            for neg_inference in neg_example[inference_type]:
                negative_emb = model.encode(neg_inference)
                cos_sim = util.cos_sim(positive_emb, negative_emb)
                # choose negatives with < 0.25 cosine similarity to gt
                if cos_sim < 0.25:
                    tokens = _split_tokens(neg_inference)
                    num_people = len([token for token in tokens if token.isdigit()])
                    # make sure there are no more person ids in the negative as in ground truth
                    if num_people <= len(person_ids):
                        return _get_inference_tokens_and_ids(neg_inference, person_ids=person_ids)

    # get negative inference from same inference type and different img using person_ids from ground truth
    def get_negative_diff_img(index, inference_type, person_ids):
        while True:
            neg_idx = random.sample([k for k in range(len(annots_dict)) if k != index], 1)[0]
            neg_example = annots_dict[neg_idx]

            if inference_type in neg_example:
                for neg_inference in neg_example[inference_type]:
                    tokens = _split_tokens(neg_inference)
                    num_people = len([token for token in tokens if token.isdigit()])
                    # make sure there are no more person ids in the negative as in ground truth
                    if num_people <= len(person_ids):
                        return _get_inference_tokens_and_ids(neg_inference, person_ids=person_ids)

    def build_mc_example(inf_type, inf_start, event_subject, event, subject_ids_set, img_fn):
        for inference in example[inf_type]:
            qa_dict = {}
            # get ground truth inference statement
            subject_w_no_obj, inference_ids_set, pos_inference = _get_inference_tokens_and_ids(
                inference, subject_ids=subject_ids_set
            )
            mc_subject = event_subject
            # if the subject of inference is not the entire subject of event
            if subject_w_no_obj != subject_ids_set:
                # let subject be the first person in remaining [set(event subj) - set(inference person ids)]
                mc_subject = '<|det%d|>' % (int(list(subject_w_no_obj)[0]))

            qa_dict['event'] = event + inf_start.format(mc_subject)
            positive_choice_idx = i % 4
            qa_dict['ending{}'.format(positive_choice_idx)] = pos_inference

            # get negative inference statements
            person_ids = list(inference_ids_set)
            for j in [k for k in range(4) if k != positive_choice_idx]:
                _, _, neg_inference = get_negative_dissimilar(inference, img_fn, inf_type, person_ids)
                qa_dict['ending{}'.format(j)] = neg_inference

            qa_dict['img_id'] = img_fn
            qa_dict['label'] = positive_choice_idx

            data_large.append(qa_dict)

    # build mc data from annotations
    data_large = []

    for i, example in enumerate(annots_dict[:100]):
        if i % 100 == 0:
            print(large_file, 'Example:', i)

        annot_event = example['event']
        img_fn = example['img_fn']
        event_subject, processed_event, subject_ids_set, event_ids_set = _get_event_tokens(annot_event)

        if 'intent' in example:
            build_mc_example('intent', '. Because {} wanted to ', event_subject, processed_event, subject_ids_set, img_fn)

        if 'before' in example:
            build_mc_example('before', '. Before {} needed to ', event_subject, processed_event, subject_ids_set, img_fn)

        if 'after' in example:
            build_mc_example('after', '. After {} will most likely ', event_subject, processed_event, subject_ids_set, img_fn)

    with open(large_file, 'w') as large_json:
        json.dump(data_large, large_json)


# read_text('../visualcomet/train_annots.json', './data/train.json')
# read_text('../visualcomet/val_annots.json', './data/val.json')
# read_text('../visualcomet/test_annots.json', './data/test2.json')

parser = argparse.ArgumentParser(description='Build MC data')

parser.add_argument('--data-src-dir', type=str)
parser.add_argument('--data-dest-dir', type=str)
args = parser.parse_args()

read_text(args.data_src_dir, args.data_dest_dir)






