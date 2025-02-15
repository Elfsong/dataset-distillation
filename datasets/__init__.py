from torchvision import datasets, transforms
from torchtext.legacy import datasets as textdata
from torchtext.legacy import data
from torchtext.vocab import GloVe
from PIL import Image
from .usps import USPS
from . import caltech_ucsd_birds
from . import pascal_voc
import os
import contextlib
import numpy as np
import torch
from collections import namedtuple
import math
import torch.nn as nn


default_dataset_roots = dict(
    MNIST='./data/mnist',
    MNIST_RGB='./data/mnist',
    SVHN='./data/svhn',
    USPS='./data/usps',
    Cifar10='./data/cifar10',
    Cifar100='./data/cifar100',
    CUB200='./data/birds',
    PASCAL_VOC='./data/pascal_voc',
    imdb='./data/text/imdb',
    sst5='./data/text/sst',
    trec6='./data/text/trec',
    trec50='./data/text/trec',
    snli='./data/text/snli',
    multinli='./data/text/multinli',
)


dataset_normalization = dict(
    MNIST=((0.1307,), (0.3081,)),
    MNIST_RGB=((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    USPS=((0.15972736477851868,), (0.25726667046546936,)),
    SVHN=((0.4379104971885681, 0.44398033618927, 0.4729299545288086),
          (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)),
    Cifar10=((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    Cifar100=((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    CUB200=((0.47850531339645386, 0.4992702007293701, 0.4022205173969269),
            (0.23210887610912323, 0.2277066558599472, 0.26652416586875916)),
    PASCAL_VOC=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    imdb=((0,),(0,)),
    sst5=((0,),(0,)),
    trec6=((0,),(0,)),
    trec50=((0,),(0,)),
    snli=((0,),(0,)),
    multinli=((0,),(0,)),
)


dataset_labels = dict(
    MNIST=list(range(10)),
    MNIST_RGB=list(range(10)),
    USPS=list(range(10)),
    SVHN=list(range(10)),
    Cifar10=('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'monkey', 'horse', 'ship', 'truck'),
    Cifar100=(
        'apple',
        'aquarium_fish',
        'baby',
        'bear',
        'beaver',
        'bed',
        'bee',
        'beetle',
        'bicycle',
        'bottle',
        'bowl',
        'boy',
        'bridge',
        'bus',
        'butterfly',
        'camel',
        'can',
        'castle',
        'caterpillar',
        'cattle',
        'chair',
        'chimpanzee',
        'clock',
        'cloud',
        'cockroach',
        'couch',
        'cra',
        'crocodile',
        'cup',
        'dinosaur',
        'dolphin',
        'elephant',
        'flatfish',
        'forest',
        'fox',
        'girl',
        'hamster',
        'house',
        'kangaroo',
        'keyboard',
        'lamp',
        'lawn_mower',
        'leopard',
        'lion',
        'lizard',
        'lobster',
        'man',
        'maple_tree',
        'motorcycle',
        'mountain',
        'mouse',
        'mushroom',
        'oak_tree',
        'orange',
        'orchid',
        'otter',
        'palm_tree',
        'pear',
        'pickup_truck',
        'pine_tree',
        'plain',
        'plate',
        'poppy',
        'porcupine',
        'possum',
        'rabbit',
        'raccoon',
        'ray',
        'road',
        'rocket',
        'rose',
        'sea',
        'seal',
        'shark',
        'shrew',
        'skunk',
        'skyscraper',
        'snail',
        'snake',
        'spider',
        'squirrel',
        'streetcar',
        'sunflower',
        'sweet_pepper',
        'table',
        'tank',
        'telephone',
        'television',
        'tiger',
        'tractor',
        'train',
        'trout',
        'tulip',
        'turtle',
        'wardrobe',
        'whale',
        'willow_tree',
        'wolf',
        'woman',
        'worm'
    ),
    CUB200=caltech_ucsd_birds.class_labels,
    PASCAL_VOC=pascal_voc.object_categories,
    imdb={0,1},
    sst5=list(range(5)),
    trec6=list(range(6)),
    trec50=list(range(50)),
    snli=list(range(3)),
    multinli=list(range(3)),
)

# (nc, real_size, num_classes)
DatasetStats = namedtuple('DatasetStats', ' '.join(['nc', 'real_size', 'num_classes']))

dataset_stats = dict(
    MNIST=DatasetStats(1, 28, 10),
    MNIST_RGB=DatasetStats(3, 28, 10),
    USPS=DatasetStats(1, 28, 10),
    SVHN=DatasetStats(3, 32, 10),
    Cifar10=DatasetStats(3, 32, 10),
    Cifar100=DatasetStats(3, 32, 100),
    CUB200=DatasetStats(3, 224, 200),
    PASCAL_VOC=DatasetStats(3, 224, 20),
    imdb = DatasetStats(1, 0, 2),
    sst5 = DatasetStats(1, 0, 5),
    trec6 = DatasetStats(1, 0, 6),
    trec50 = DatasetStats(1, 0, 50),
    snli = DatasetStats(1, 0, 3),
    multinli = DatasetStats(1, 0, 3),
)

assert(set(default_dataset_roots.keys()) == set(dataset_normalization.keys()) ==
       set(dataset_labels.keys()) == set(dataset_stats.keys()))

def print_closest_words(vec, glove, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[0:n+1]: 					       # take the top n
        print(glove.itos[idx], difference)
def closest_words(vec, glove, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    return [glove.itos[idx] for idx, _ in lst[0:n+1]]				       # take the top n
def get_info(state):
    dataset_stats['imdb']=DatasetStats(1,state.maxlen,2)
    dataset_stats['sst5']=DatasetStats(1,state.maxlen,5)
    dataset_stats['trec6']=DatasetStats(1,state.maxlen,6)
    dataset_stats['trec50']=DatasetStats(1,state.maxlen,50)
    dataset_stats['snli']=DatasetStats(1,state.maxlen,3)
    dataset_stats['multinli']=DatasetStats(1,state.maxlen,3)
    name = state.dataset  # argparse dataset fmt ensures that this is lowercase and doesn't contrain hyphen
    assert name in dataset_stats, 'Unsupported dataset: {}'.format(state.dataset)
    nc, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    root = state.dataset_root
    if root is None:
        root = default_dataset_roots[name]
    labels = dataset_labels[name]
    return name, root, nc, input_size, num_classes, normalization, labels


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


def get_dataset(state, phase):
    dataset_stats['imdb']=DatasetStats(1,state.maxlen,2)
    dataset_stats['sst5']=DatasetStats(1,state.maxlen,5)
    dataset_stats['trec6']=DatasetStats(1,state.maxlen,6)
    dataset_stats['trec50']=DatasetStats(1,state.maxlen,50)
    dataset_stats['snli']=DatasetStats(1,state.maxlen,3)
    dataset_stats['multinli']=DatasetStats(1,state.maxlen,3)
    assert phase in ('train', 'test'), 'Unsupported phase: %s' % phase
    name, root, nc, input_size, num_classes, normalization, _ = get_info(state)
    real_size = dataset_stats[name].real_size

    if name == 'MNIST':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'MNIST_RGB':
        transform_list = [transforms.Grayscale(3)]
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == 'train'), download=True,
                                  transform=transforms.Compose(transform_list))
    elif name == 'USPS':
        if input_size != real_size:
            transform_list = [transforms.Resize([input_size, input_size], Image.BICUBIC)]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return USPS(root, train=(phase == 'train'), download=True,
                        transform=transforms.Compose(transform_list))
    elif name == 'SVHN':
        transform_list = []
        if input_size != real_size:
            transform_list.append(transforms.Resize([input_size, input_size], Image.BICUBIC))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.SVHN(root, split=phase, download=True,
                                 transform=transforms.Compose(transform_list))
    elif name == 'Cifar10':
        transform_list = []
        if input_size != real_size:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        if phase == 'train':
            transform_list += [
                # TODO: merge the following into the padding options of
                #       RandomCrop when a new torchvision version is released.
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.CIFAR10(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'Cifar100':
        transform_list = []
        if input_size != real_size:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        if phase == 'train':
            transform_list += [
                # TODO: merge the following into the padding options of
                #       RandomCrop when a new torchvision version is released.
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        with suppress_stdout():
            return datasets.CIFAR100(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'CUB200':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        return caltech_ucsd_birds.CUB200(root, phase == 'train', transforms.Compose(transform_list), download=True)
    elif name == 'PASCAL_VOC':
        transform_list = []
        if phase == 'train':
            transform_list += [
                transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list += [
                transforms.Resize([input_size, input_size], Image.BICUBIC),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
        if phase == 'train':
            phase = 'trainval'
        return pascal_voc.PASCALVoc2007(root, phase, transforms.Compose(transform_list))
    elif name == 'imdb':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.IMDB.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #man=TEXT.vocab.vectors[TEXT.vocab["man"]].clone()
        #woman=TEXT.vocab.vectors[TEXT.vocab["woman"]].clone()
        #king=TEXT.vocab.vectors[TEXT.vocab["doctor"]].clone()
        
        #print(torch.norm(king - man + woman))
        #vec = king - man + woman
        #print_closest_words(vec, TEXT.vocab)
        #print_closest_words(king, TEXT.vocab)
        #print(TEXT.vocab.vectors)
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'sst5':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.SST.splits(TEXT, LABEL, fine_grained=True)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'trec6':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.TREC.splits(TEXT, LABEL, fine_grained=False)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'trec50':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, test = textdata.TREC.splits(TEXT, LABEL, fine_grained=True)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'snli':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.SNLI.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src
    elif name == 'multinli':
        transform_list = []
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=state.maxlen)
        LABEL = data.LabelField(dtype=torch.long)
        
        # make splits for data
        train, valid, test = textdata.MultiNLI.splits(TEXT, LABEL)
        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=state.ninp, max_vectors=state.ntoken), max_size=state.ntoken-2) #max_size=state.ntoken,
        LABEL.build_vocab(train)
        #print(len(TEXT.vocab))
        #print(len(LABEL.vocab))
        state.pretrained_vec=TEXT.vocab.vectors
        state.glove = TEXT.vocab
        #ninp=32 #Maybe 400
        #ntoken=32
        #encoder = nn.Embedding(ntoken, ninp)
        
        #train_iter, test_iter = textdata.IMDB.iters(batch_size=state.batch_size, fix_length=state.ninp)
        if phase=="train":
            src=train
            #src = encoder(train_iter) * math.sqrt(ninp)
        else:
            src=test
            #src = encoder(test_iter) * math.sqrt(ninp)
            
        #src = data.Iterator.splits(
        #src, batch_size=state.batch_size, device=state.device, repeat=False, sort_key=lambda x: len(x.src))
        
        return src

    else:
        raise ValueError('Unsupported dataset: %s' % state.dataset)
