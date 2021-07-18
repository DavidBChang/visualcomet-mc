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







