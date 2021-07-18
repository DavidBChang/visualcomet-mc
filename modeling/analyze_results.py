import json
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import copy
import clip
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import RobertaTokenizerFast, RobertaForMultipleChoice, AdamW
from tqdm import tqdm
from pandas import DataFrame
import os
from dataclasses import dataclass


torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


parser = argparse.ArgumentParser(description='Text-Only Examples')

parser.add_argument('--batch-size', type=int, default=32, metavar='N')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--train-size', type=int, default=100)
args = parser.parse_args()

def read_text(path, train_flag=False):
    """
    Reads data from a json file and prepares datasets for sequence classification.
    Returns large and small-sized datasets.
    """
    with open(Path(path), 'rb') as f:
        data = json.load(f)

    prompts = []
    answers = []
    labels = []

    if train_flag and args.train_size >= 2:
        data = data[:len(data) // args.train_size]

    for i, example in enumerate(data):
        if i % 1000 == 0:
            print(path, 'Example:', i)
        prompts.append([example['event']] * 4)
        answers.append([example['ending0'], example['ending1'], example['ending2'], example['ending3']])
        labels.append(example['label'])

    return prompts, answers, labels


prompts_test, answers_test, labels_test = read_text('../data/test.json')


class Text_MC_Dataset(torch.utils.data.Dataset):
    def __init__(self, prompts, answers, labels):
        self.prompts = prompts
        self.answers = answers
        self.labels = labels
        self.tokenizer = RobertaTokenizerFast.from_pretrained(args.model)

    def __getitem__(self, idx):
        # prompt = self.prompts[idx]      # [p, p, p, p]
        # answer = self.answers[idx]      # [a1, a2, a3, a4]
        # encodings = self.tokenizer(prompt, answer, padding=True, truncation=True, return_tensors="pt")
        # item = {k: v.unsqueeze(0) for k, v in encodings.items()}
        item = {}
        item['prompt'] = self.prompts[idx]
        item['answer'] = self.answers[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        # print(item['input_ids'].shape)        # (1, 4, N=33,34,...)
        # print(item['attention_mask'].shape)
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    # tokenizer: PreTrainedTokenizerBase
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    def __init__(self, tokenizer=RobertaTokenizerFast.from_pretrained(args.model)):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # print('first item:', batch[0])
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
        # print('entire batch:', new_batch['input_ids'].shape)
        return new_batch


large_test_dataset = Text_MC_Dataset(prompts_test, answers_test, labels_test)

large_test_loader = DataLoader(
    large_test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=DataCollatorForMultipleChoice()
)
batch_size = 32


def test(test_loader, best_model, path):
    with open(Path(path), 'rb') as f:
        data = json.load(f)

    table = []
    num_rows = 0
    correct = 0
    running_loss = 0
    num_seqs = 0
    best_model.eval()
    tqdm_test_loader = tqdm(test_loader)

    True_Pos = 0
    False_Pos = 0
    True_Neg = 0
    False_Neg = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            logits = logits.detach()
            labels = labels.detach()

            loss = F.cross_entropy(logits, labels)

            # Compute loss and number of correct predictions and accumulate metrics and update status
            running_loss += loss.detach().item()
            correct += torch.sum(torch.eq(torch.argmax(F.softmax(logits, dim=1), dim=1), labels))
            num_seqs += len(labels)
            tqdm_test_loader.set_description_str(
                f"[Loss]: {running_loss / (batch_idx + 1):.4f} [Acc]: {correct / num_seqs:.4f}")
            # print('logits:', logits)
            predicted = torch.argmax(F.softmax(logits, dim=1), dim=1)
            # print('predicted:', predicted)
            # print('labels:', labels)
            if num_rows < 5000:
                for i in range(batch_size):
                    idx = batch_idx * batch_size + i
                    event = data[idx]['event']
                    c0, c1, c2, c3 = (data[idx]['ending0'], data[idx]['ending1'],
                                      data[idx]['ending2'], data[idx]['ending3'])
                    label_i = labels[i].item()
                    predicted_i = predicted[i].item()
                    # if label_i == 1 and predicted_i == 1:
                    #     True_Pos += 1
                    # elif label_i == 0 and predicted_i == 1:
                    #     False_Pos += 1
                    # elif label_i == 0 and predicted_i == 0:
                    #     True_Neg += 1
                    # else:
                    #     False_Neg += 1

                    example = [event, c0, c1, c2, c3, label_i, predicted_i]
                    table.append(example)

            num_rows += batch_size

        loss_test = running_loss / (batch_idx + 1)

        df = DataFrame(table,
                       columns=['event', 'choice1', 'choice2', 'choice3', 'choice4', 'ground truth', 'predicted'])
        # df.style.apply(highlight_incorrect, axis=1)
        html = create_html_table(df)
        # html = df.to_html(classes='table table-striped')

        # write html to file
        text_file = open("./visualizations/mc_text.html", "w")
        text_file.write(html)
        text_file.close()

        # precision = True_Pos / (True_Pos + False_Pos)
        # recall = True_Pos / (True_Pos + False_Neg)
        # npv = True_Neg / (True_Neg + False_Neg)
        # tnr = True_Neg / (True_Neg + False_Pos)

    print('Accuracy of the network on the test sequences: %.2f %%' % (100 * correct / num_seqs))
    print('Loss of the network on the test sequences: {}'.format(loss_test))
    # print('True positives:', True_Pos, '; False positives:', False_Pos,
    #       '; True negatives:', True_Neg, '; False negatives:', False_Neg)
    # print('precision:', precision, 'recall:', recall, 'negative predictive value:', npv, 'specificity:', tnr)


def create_html_table(x):
    row_data = '''
    <html>
      <head><title>HTML Pandas Dataframe with CSS</title></head>
      <link rel="stylesheet" type="text/css" href="df_style.css"/>
      <body>
      <table border="1" class="dataframe table table-striped">
      <thead>
        <tr style="text-align: center;">
          <th></th>
          <th>Event</th>
          <th>Choice0</th>
          <th>Choice1</th>
          <th>Choice2</th>
          <th>Choice3</th>
          <th>ground truth</th>
          <th>predicted</th>
        </tr>
      </thead>
      <tbody>
    '''

    for i in range(x.shape[0]):
        if x.iloc[i]['ground truth'] == x.iloc[i]['predicted']:
            row_data += '\n<tr style="background-color: white"> \n <th>' + str(i) + '</th>'
            for j in range(x.shape[1]):
                row_data += '\n<td>' + str(x.iloc[i, j]) + '</td>'
            row_data += '\n</tr>'
        else:
            row_data += '\n<tr> \n <th>' + str(i) + '</th>'
            row_data += '\n<td>' + str(x.iloc[i, 0]) + '</td>'
            for j in range(1, x.shape[1]):
                if j - 1 == x.iloc[i]['ground truth']:
                    row_data += '\n<td style="background-color: lightgreen">' + str(x.iloc[i, j]) + '</td>'
                elif j - 1 == x.iloc[i]['predicted']:
                    row_data += '\n<td style="background-color: pink">' + str(x.iloc[i, j]) + '</td>'
                else:
                    row_data += '\n<td>' + str(x.iloc[i, j]) + '</td>'
            row_data += '\n</tr>'

    row_data += '\n </tbody>\n </table>\n </body>\n </html>'

    return row_data


best_model_path = '../models/text_mc.pth'

model = RobertaForMultipleChoice.from_pretrained(args.model)
model.load_state_dict(torch.load(best_model_path))
model.to(device)

test(large_test_loader, model, '../data/test.json')



