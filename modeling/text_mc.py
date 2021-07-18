import json
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import RobertaTokenizerFast, RobertaForMultipleChoice, AdamW
from tqdm import tqdm
from dataclasses import dataclass
import random


parser = argparse.ArgumentParser(description='Text-Only Examples')

parser.add_argument('--batch-size', type=int, default=32, metavar='N')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--train-size', type=int, default=2)
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

    for i, example in enumerate(data):  # [:len(data) // 100]
        if i % 1000 == 0:
            print(path, 'Example:', i)

        prompts.append([example['event'].split('.')[-1]] * 4)
        answers.append([example['ending0'], example['ending1'], example['ending2'], example['ending3']])
        labels.append(example['label'])

    return prompts, answers, labels


prompts_train, answers_train, labels_train = read_text('../data/train.json', True)
prompts_val, answers_val, labels_val = read_text('../data/val.json')
# prompts_test, answers_test, labels_test = read_text('../data/test.json')


class Text_MC_Dataset(torch.utils.data.Dataset):
    def __init__(self, prompts, answers, labels):
        self.prompts = prompts
        self.answers = answers
        self.labels = labels

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


large_train_dataset = Text_MC_Dataset(prompts_train, answers_train, labels_train)
large_val_dataset = Text_MC_Dataset(prompts_val, answers_val, labels_val)
# large_test_dataset = Text_MC_Dataset(prompts_test, answers_test, labels_test)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('\tUsing', torch.cuda.device_count(), 'GPUs')
print('\tbatch size:', args.batch_size)
print('\tlr:', args.lr)
print('\tmodel:', args.model)
print('\ttraining size:', args.train_size)
model = RobertaForMultipleChoice.from_pretrained(args.model)
model = nn.DataParallel(model)
model.to(device)

large_train_loader = DataLoader(
    large_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=DataCollatorForMultipleChoice()
)
large_val_loader = DataLoader(
    large_val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=DataCollatorForMultipleChoice()
)

optim = AdamW(model.parameters(), lr=args.lr)
NUM_EPOCHS = 3


def train(model, optimizer, train_loader, val_loader, num_epochs, best_model_path):
    best_accuracy_val = 0
    best_accuracy_train = 0
    losses_train = []
    losses_val = []

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
            # token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optim.step()

            # Accumulate metrics and update status
            train_loss += loss.detach().item()
            # print('logits:', logits)
            train_correct += (torch.sum(torch.eq(torch.argmax(F.softmax(logits.detach(), dim=1), dim=1), labels)))
            train_seqs += len(labels)
            tqdm_train_loader.set_description_str(
                f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}")
        print()

        avg_train_loss = train_loss / len(tqdm_train_loader)
        losses_train.append(avg_train_loss)
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
                # token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                logits = outputs.logits
                logits = logits.detach()
                labels = labels.detach()

                loss = F.cross_entropy(logits, labels)

                # Compute loss and number of correct predictions and accumulate metrics and update status
                val_loss += loss.detach().item()
                val_correct += (torch.sum(torch.eq(torch.argmax(F.softmax(logits, dim=1), dim=1), labels)))
                val_seqs += len(labels)
                tqdm_val_loader.set_description_str(
                    f"[Loss]: {val_loss / (batch_idx + 1):.4f} [Acc]: {val_correct / val_seqs:.4f}")
        print()

        avg_val_loss = val_loss / len(tqdm_val_loader)
        losses_val.append(avg_val_loss)
        val_accuracy = val_correct / val_seqs
        print(f"[Validation Loss]: {avg_val_loss:.4f} [Validation Accuracy]: {val_accuracy:.4f}")
        # find best training and validation accuracy,
        if train_accuracy > best_accuracy_train:
            best_accuracy_train = train_accuracy
        if val_accuracy > best_accuracy_val:
            best_accuracy_val = val_accuracy
            best_model_state = copy.deepcopy(model.module.state_dict())
            torch.save(best_model_state, best_model_path)

    print('Finished Training')
    print(f"[Best Training Accuracy]: {best_accuracy_train:.4f}")
    print(f"[Best Validation Accuracy]: {best_accuracy_val:.4f}")

    return losses_train, losses_val


def test(test_loader, best_model):
    correct = 0
    running_loss = 0
    num_seqs = 0
    best_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            # predicted = torch.argmax(F.softmax(logits, dim=1), dim=1)
            # for i, eq in enumerate(torch.eq(predicted, labels)):
            #     if not eq:
            #         idx = batch_idx * args.batch_size + i
            #         print('At index', idx)
            #         print('Incorrect:', data[idx])
            #         print('Expected:', labels[i].item(), 'Actual:', predicted[i].item())
            #         print()
            #         break

            loss = F.cross_entropy(logits, labels)
            running_loss += loss.item()

            num_seqs += len(labels)
            correct += (torch.sum(torch.eq(torch.argmax(F.softmax(logits, dim=1), dim=1), labels)))
        loss_test = running_loss / (batch_idx + 1)

    print('Accuracy of the network on the test sequences: %.2f %%' % (100 * correct / num_seqs))
    print('Loss of the network on the test sequences: {}'.format(loss_test))


def analyze_results(test_loader, best_model, path):
    with open(Path(path), 'rb') as f:
        data = json.load(f)

    best_model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        batch = dataiter.next()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predicted = torch.argmax(F.softmax(logits, dim=1), dim=1)
        for i, eq in enumerate(torch.eq(predicted, labels)):
            if not eq:
                print('At index', i)
                print('Incorrect:', data[i])
                print('Expected:', labels[i].item(), 'Actual:', predicted[i].item())


best_model_path = '../models/text_mc.pth'
losses_train, losses_val = train(model, optim, large_train_loader, large_val_loader, NUM_EPOCHS, best_model_path)

# model = RobertaForMultipleChoice.from_pretrained(args.model)
# model.load_state_dict(torch.load(best_model_path))
# model.to(device)

# test(large_test_loader, model)
# analyze_results(large_test_loader, model, './data/same_img_negs/test_large.json')






