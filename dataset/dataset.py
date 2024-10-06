import torch.utils.data

# TODO

class ImageDataset( torch.utils.data.Dataset ):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def __getitem__(self, i: int):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
