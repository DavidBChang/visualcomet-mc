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

    for i, example in enumerate(annots_dict):
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


read_text('../visualcomet/train_annots.json', './data/train.json')
read_text('../visualcomet/val_annots.json', './data/val.json')
# read_text('../visualcomet/test_annots.json', './data/test2.json')

# parser = argparse.ArgumentParser(description='Build MC data')
#
# parser.add_argument('--data-src-dir', type=str)
# parser.add_argument('--data-dest-dir', type=str)
# args = parser.parse_args()
#
# read_text(args.data_src_dir, args.data_dest_dir)


