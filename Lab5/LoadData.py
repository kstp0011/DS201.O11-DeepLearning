import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import underthesea
from collections import Counter

class UITvsfc_dataset(Dataset):
    def __init__(self, path, vocab=None):
        super().__init__()
        data = pd.read_csv(path)
        self.sentences = [underthesea.word_tokenize(
            sentence) for sentence in data['sentence'].values]
        self.label = torch.from_numpy(data['sentiment'].values)

        if vocab is None:
            words = [word for sentence in self.sentences for word in sentence]
            word_counts = Counter(words)
            self.vocab = {word: i+1 for i,
                          (word, _) in enumerate(word_counts.most_common())}
            self.vocab['<PAD>'] = 0
        else:
            self.vocab = vocab

        self.sentences = [self.numericalize_and_pad(
            sentence) for sentence in self.sentences]

    def numericalize_and_pad(self, sentence):
        numericalized = [self.vocab.get(
            word, self.vocab['<PAD>']) for word in sentence]
        return torch.tensor(numericalized)

    def __getitem__(self, index):
        return self.sentences[index], self.label[index]

    def __len__(self):
        return len(self.label)


class dataset_loader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        sentences = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])

        # Find the length of the longest sentence in the batch
        max_length = max([len(sentence) for sentence in sentences])

        # Pad all sentences to the length of the longest sentence
        sentences = [F.pad(sentence, pad=(0, max_length - len(sentence)))
                     for sentence in sentences]

        # Stack the sentences
        sentences = torch.stack(sentences)

        return sentences, labels