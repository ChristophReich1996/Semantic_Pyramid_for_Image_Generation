import torch
from torch.utils.data import DataLoader
import kornia

from data import Places365, image_label_list_of_masks_collate_function
from models import VGG16

if __name__ == '__main__':
    # Init dataset
    training_dataset = DataLoader(
        Places365(path_to_index_file='/home/creich/places365_standard', index_file_name='train.txt'),
        batch_size=10, num_workers=10, shuffle=True, drop_last=True,
        collate_fn=image_label_list_of_masks_collate_function)
    # Init VGG
    vgg16 = VGG16(return_output=True)
    vgg16.load_state_dict(torch.load('vgg_places365_fine_tuned.pt', map_location="cpu"))
    vgg16.eval()
    # Get dataset samples
    images, labels, _ = next(iter(training_dataset))
    images = kornia.normalize(images,
                              mean=torch.tensor([0.485, 0.456, 0.406], device=images.device),
                              std=torch.tensor([0.229, 0.224, 0.225], device=images.device))
    # Make prediction
    predictions = vgg16(images)
    print(predictions.softmax(dim=-1).argmax(dim=-1))
    print(torch.topk(predictions, k=10, dim=-1)[1])
    print(labels.argmax(dim=-1))
