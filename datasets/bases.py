import os
from typing import List

import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from transformers import CLIPTokenizer

from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
            self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
            self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 test_mode='dense',
                 truncate: bool = True,
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.test_mode = test_mode
        # print(self.test_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        img_path, image_id, _, caption = self.dataset[index]
        tokens = tokenize(caption[0],
                          tokenizer=self.tokenizer,
                          text_length=self.text_length,
                          truncate=self.truncate)
        if self.test_mode == 'rss':
            return self.random_seq(img_path, image_id, tokens)
        elif self.test_mode == 'dense':
            # print("-----",self.test_mode)
            return self.dense_seq(img_path, image_id, tokens)
        else:
            # Handle other sampling methods here
            pass

    def random_seq(self, img_path, image_id, tokens):
        img_path = np.array(list(img_path))
        # Process images
        img = [read_image(img_paths) for img_paths in img_path]
        seq = [img]

        if self.transform is not None:
            seq = self.transform(seq)
        ret = {
            'pids': image_id,
            'images': torch.stack(seq[0], dim=0),  # seq_len x channels x height x width
            'caption': tokens,
        }
        return ret

    def dense_seq(self, img_path, image_id, token):
        seqs = img_path
        process_seq = []
        pids = []
        tokens = []
        for s in seqs:
            ret = self.random_seq(s, image_id, token)
            process_seq.append(ret['images'])
            pids.append(image_id)
            tokens.append(token)

        stacked_imgs = torch.stack(process_seq, dim=0)
        pids_tensor = torch.tensor(pids).unsqueeze(1)

        tokens_tensor = torch.stack(tokens)
        ret = {
            'pids': pids_tensor,
            'images': stacked_imgs,  # seq_len x channels x height x width
            'caption': tokens_tensor,
        }
        return ret

class ImageValTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 test_mode='dense',
                 truncate: bool = True,
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.test_mode = test_mode
        # print(self.test_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        img_path, image_id, _, caption = self.dataset[index]
        tokens = tokenize(caption[0],
                          tokenizer=self.tokenizer,
                          text_length=self.text_length,
                          truncate=self.truncate)
        if self.test_mode == 'rss':
            return self.random_seq(img_path, image_id, tokens,caption)
        elif self.test_mode == 'dense':
            # print("-----",self.test_mode)
            return self.dense_seq(img_path, image_id, tokens,caption)
        else:
            # Handle other sampling methods here
            pass

    def random_seq(self, img_path, image_id, tokens,caption):
        copy_img_path = list(img_path)
        img_path = np.array(list(img_path))
        img = [read_image(img_paths) for img_paths in img_path]
        seq = [img]

        if self.transform is not None:
            seq = self.transform(seq)

        ret = {
            'pids': image_id,
            'images': torch.stack(seq[0], dim=0),  # seq_len x channels x height x width
            'caption': tokens,
            'img_paths': copy_img_path,
            'cname': caption
        }
        return ret

    def dense_seq(self, img_path, image_id, token,caption):
        seqs = img_path
        process_seq = []
        pids = []
        tokens = []
        # cname = []
        for s in seqs:
            ret = self.random_seq(s, image_id, token)
            process_seq.append(ret['images'])
            pids.append(image_id)
            tokens.append(token)

        stacked_imgs = torch.stack(process_seq, dim=0)
        # 将 pids 转换为 (batchsize, 1) 的张量
        pids_tensor = torch.tensor(pids).unsqueeze(1)  # 添加维度变成 (batchsize, 1)

        # 将 tokens 直接 stack
        tokens_tensor = torch.stack(tokens)
        ret = {
            'pids': pids_tensor,
            'images': stacked_imgs,  # seq_len x channels x height x width
            'caption': tokens_tensor,
        }
        return ret
class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 test_mode='rss'):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.test_mode = test_mode

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        # pid, image_id, img_path, caption,_ = self.dataset[index]
        img_path, image_id, _, caption = self.dataset[index]
        img_path = np.array(list(img_path))
        img = [read_image(img_paths) for img_paths in img_path]
        seq = [img]
        if self.transform is not None:
            seq = self.transform(seq)

        caption_tokens = tokenize(caption[0], tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        # print(caption_tokens.shape)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': image_id,
            'images': torch.stack(seq[0], dim=0),
            'caption': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)