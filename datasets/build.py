import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler

from utils.comm import get_world_size
from .Tmars import Tmars

from .bases import ImageTextDataset, ImageTextMLMDataset, ImageValTextDataset

from utils import seqtransforms as SeqT
__factory = {'Tmars': Tmars}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])

    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            batch_tensor_dict.update({k:v})

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("Tmars.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir,seq_len=args.seq_len,mode=args.test_mode)

    num_classes = dataset.num_train_pids

    if args.training:
        train_transforms = SeqT.Compose([SeqT.RectScale(args.img_size[0], args.img_size[1]),
                                        SeqT.RandomHorizontalFlip(),
                                        SeqT.RandomSizedEarser(),
                                        SeqT.ToTensor(),
                                        SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_transforms = SeqT.Compose([SeqT.RectScale(args.img_size[0], args.img_size[1]),
                                       SeqT.ToTensor(),
                                       SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length,test_mode='rrs')
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length,test_mode='rrs')

        if args.sampler == 'identity':
            logger.info(
                f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
            )
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=RandomIdentitySampler(
                                          dataset.train, args.batch_size,
                                          args.num_instance),
                                      num_workers=num_workers,
                                      collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set

        val_set = ImageTextDataset(dataset.query+dataset.gallery,
                                     val_transforms,
                                     text_length=args.text_length,test_mode=args.test_mode)

        val_loader = DataLoader(val_set,
                                  batch_size=args.batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate)

        return train_loader, val_loader, num_classes

    else:
        # build dataloader for testing

        val_transforms = SeqT.Compose([SeqT.RectScale(args.img_size[0], args.img_size[1]),
                                       SeqT.ToTensor(),
                                       SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        #if visualization use ImageValTextDataset
        val_set = ImageTextDataset(dataset.query + dataset.gallery,
                                   val_transforms,
                                   text_length=args.text_length,test_mode=args.test_mode)


        val_loader = DataLoader(val_set,
                                batch_size=args.batch_size, #todo dense set to 1
                                num_workers=num_workers,
                                collate_fn=collate)

        return val_loader, num_classes
