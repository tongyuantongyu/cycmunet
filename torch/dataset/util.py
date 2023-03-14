import bisect

import torch.utils.data as data


class InterleavedDataset(data.Dataset):
    def __init__(self, *datasets: data.Dataset):
        self.datasets = datasets
        if not all(hasattr(i, '__len__') for i in datasets):
            raise AttributeError('need datasets with known length')
        sizes = [len(i) for i in datasets]
        self.total = sum(sizes)
        self.sizes = [0] + sorted(set(sizes))
        self.index = {
            0: datasets
        }
        total, last_n = 0, len(datasets)
        for last_size, size in zip(self.sizes, self.sizes[1:]):
            total += (size - last_size) * last_n
            this_datasets = [ds for ds in datasets if len(ds) > size]
            self.index[total] = this_datasets
            last_n = len(this_datasets)
        self.index.popitem()
        self.index_keys = list(self.index.keys())

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        stage = bisect.bisect_right(self.index_keys, idx) - 1
        offset = self.sizes[stage]
        begin = self.index_keys[stage]
        idx -= begin
        datasets = self.index[begin]
        n, i = idx % len(datasets), idx // len(datasets)
        return datasets[n][offset + i]
