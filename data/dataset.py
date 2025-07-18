r""" Dataloader builder """
from torch.utils.data import DataLoader

from data.crack import DatasetCRACK
from data.drive import DatasetDRIVE
from data.roads import DatasetROAD

class CSDataset:

    @classmethod
    def initialize(cls, datapath):

        cls.datasets = {
            'cracktree200': DatasetCRACK,
            'crack500': DatasetCRACK,
            'cracktav': DatasetCRACK,
            'crackls315': DatasetCRACK,
            'deepcrack': DatasetCRACK,
            'tc': DatasetCRACK,
            'drive': DatasetDRIVE,
            'roads': DatasetROAD,
        }

        cls.datapath = datapath


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, img_mode, img_size):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'train'
        nworker = nworker #if split == 'trn' else 0

        if split == 'train':
            dataset = cls.datasets[benchmark](benchmark,
                                              datapath=cls.datapath,
                                              split=split,
                                              img_mode=img_mode,
                                              img_size=img_size)
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)
        else:
            dataset = cls.datasets[benchmark](benchmark,
                                              datapath=cls.datapath,
                                              split=split,
                                              img_mode='same',
                                              img_size=None)
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=False)



        return dataloader
