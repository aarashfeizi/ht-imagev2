from torchvision.datasets import ImageNet as ImageNetT
from typing import Any, Dict, List, Iterator, Optional, Tuple

class ImageNet(ImageNetT):
    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, number_of_classes=None, specific_classes=None, **kwargs: Any) -> None:
        super(ImageNet, self).__init__(root=root, split=split, **kwargs)
        class_threshold = number_of_classes

        if class_threshold is not None \
            and class_threshold != 0:
            new_samples = []
            for path, label in self.samples:
                if label < class_threshold:
                    new_samples.append((path, label))
            
            print(f'changed number of classes from 1000 to {class_threshold}')
            print(f'changed number of samples from {len(self.samples)} to {len(new_samples)}')
            self.samples = new_samples
        
        if specific_classes is not None \
            and len(specific_classes) != 0:
            self.samples = new_samples
            new_samples = []
            class_indexes = [self.wnid_to_idx[clss] for clss in specific_classes]
            print(f'Only choosing classes {specific_classes}')
            print(f'Using class indices {class_indexes}')
            for path, label in self.samples:
                if label in class_indexes:
                    new_samples.append((path, label))

            print(f'changed number of classes from 1000 to {len(specific_classes)}')
            print(f'changed number of samples from {len(self.samples)} to {len(new_samples)}')
            self.samples = new_samples
            