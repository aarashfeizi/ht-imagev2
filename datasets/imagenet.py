from torchvision.datasets import ImageNet as ImageNetT
from typing import Any, Dict, List, Iterator, Optional, Tuple

class ImageNet(ImageNetT):
    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(root, split, download, **kwargs)
        class_threshold = kwargs.get('number_of_classes', None)
        if class_threshold is not None \
            and class_threshold != 0:
            new_samples = []
            for path, label in self.samples:
                if label < class_threshold:
                    new_samples.append((path, label))
            
            print(f'changed number of classes from 1000 to {class_threshold}')
            print(f'changed number of samples from {len(self.samples)} to {len(new_samples)}')
            self.samples = new_samples
            