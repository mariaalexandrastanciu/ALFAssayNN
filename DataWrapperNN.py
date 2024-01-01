# Created by alexandra at 08/09/2023
# Created by alexandra at 28/07/2023
from torch.utils.data import Dataset, DataLoader
import torch as t
import numpy as np
import CONSTANTS as c
# (PatientId, Label, ctDNADetected, VAFg0p001) = range(4)


def dataWrapper(batch_size, num_workers, data, shuffle=True):
    dataset = MultiTaskDataset(data)
    data_generator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=None,
                                num_workers=num_workers, collate_fn=multiTaskCollate)
    return data_generator


def batch_padding(dataset, batch_size):
    """
    function used to add random samples from dataset to have complete batches
    params: dataset(ndarray) - the dataset to be separates in batches
            batch_size(int) - model parameter representing the batch size
    return: dataset with (batch_size - len(dataset) % batch_size) number of samples added
    """
    mod = len(dataset) % batch_size
    if mod > 0:
        nr_of_samples = batch_size - mod
        for i in range(nr_of_samples):
            rnd = np.random.randint(len(dataset) - 1)
            # dataset = np.append(dataset, dataset[rnd])
            dataset.append(dataset[rnd])
    return dataset

def multiTaskCollate(batch):
    # TODO:
    nr_samples = len(batch)
    # print(batch)
    fragmentSize = []
    labels = []
    ctDNADetection = []
    ctDNAbyVAF = []
    sampleNames = []
    for sample in batch:
        # sampleNames += sample[0]
        labels += [sample[1][1]]
        ctDNADetection += [sample[1][2]]
        ctDNAbyVAF += [sample[1][3]]
        sampleNames += [sample[1][0]]
        fragmentSize += [t.from_numpy(np.array(sample[0], dtype=np.float64))]
    fragmentSize = t.stack(fragmentSize)
    labels = t.stack(labels)
    ctDNADetection = t.stack(ctDNADetection)
    ctDNAbyVAF = t.stack(ctDNAbyVAF)

    return fragmentSize, sampleNames,  labels, ctDNADetection, ctDNAbyVAF

class MultiTaskDataset(Dataset):

    def __init__(self, x):
        self.fragmentSizes = []
        self.Y = []
        # tasks = x[1]
        # fragmentsFeatures = x[0]
        ## this function return two arrays, one with the unique sample name and another with the indexes, I am
        ## interested in the indexes, but also I am removing the last index, as for some reason besides the sample
        ## names I have also "PatientId";I do tasks[:, 0] because the unique function works only forone column
        # y_indexes = np.unique(tasks[:, PatientId], return_index=True)[1][:-1]
        for sample in x:
            sampleName = [sample[1][c.PatientId]]
            label = t.tensor(sample[1][c.Label], dtype=t.float64)
            ctDNADetection = t.tensor(sample[1][c.ctDNADetected], dtype=t.float64)
            ctDNAbyVAF = t.tensor(sample[1][c.VAFg0p001], dtype=t.float64)

            # if sample[1][Label] == "Cancer":
            #     label = t.tensor([1], dtype=t.float64)
            # else:
            #     label = t.tensor([0],  dtype=t.float64)

            # if sample[1][ctDNADetected]==True:
            #     ctDNADetection = t.tensor([1], dtype=t.float64)
            # elif sample[1][ctDNADetected] == False:
            #     ctDNADetection = t.tensor([0],  dtype=t.float64)
            # else:
            #     ctDNADetection = t.tensor([-1], dtype=t.float64)
            #
            # if sample[1][VAFg0p001] == True:
            #     ctDNAbyVAF = t.tensor([1], dtype=t.float64)
            # elif sample[1][VAFg0p001] == False:
            #     ctDNAbyVAF = t.tensor([0], dtype=t.float64)
            # else:
            #     ctDNAbyVAF = t.tensor([-1], dtype=t.float64)

            
            # vaf = t.tensor([sample[1][VAF]], dtype=t.float64)
            # fragmentsFeaturesIndexes = np.where(tasks[:, PatientId] == sampleName)
            # self.fragmentSizes += [t.from_numpy(np.vstack(fragmentsFeatures[fragmentsFeaturesIndexes]).
            #                                     astype(np.float))]
            self.fragmentSizes += [sample[0]]
            self.Y += [[sampleName, label, ctDNADetection, ctDNAbyVAF]]

    def __len__(self):
        return len(self.fragmentSizes)

    def __getitem__(self, idx):
        return self.fragmentSizes[idx], self.Y[idx]
