from maskrcnn.structures.image_list import ImageList
from model.data.utils import normalize_image
from captum.metrics import infidelity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import cv2


device = torch.device('cuda:0')


def perturb_fn(inputs):
    '''Returns generated perturbation and perturbed images
        Args:
            inputs: tensor - Input image'''
    
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape), device=device).float()
    return noise, inputs - noise


def compute_infidelity(images_list, attributes, perturb_fn, forward_fn, target):
    '''Computes infidelity for the given list of images, calculated attributes according to the generated
        perturbations
        Args:
            images_list: list[tensor] or list[ImageList] - Input images for which attributes are calculated
            attributes: tensor or ndarray - Calculated attributes 
            perturb_fn: callable - Function that returns perturbations and perturbed inputs
            forward_fn: callable - The forward function of the model or any modification of it
            target: int or tensor - Indices for selecting predictions from output'''
    
    infidelity_list = []
    bsz = images_list[0].tensors.shape[0]
    
    for image, attr in zip(images_list, attributes):
        if isinstance(image, ImageList):  # ImageList's shape is (3, C, H, W)
            image = image.tensors
            
        if isinstance(attr, np.ndarray) and len(attr.shape) != len(image.shape):
            attr = torch.from_numpy(attr)
            attr = torch.tile(attr.unsqueeze(0), (3, 1, 1))
            attr = torch.stack((attr, attr, attr), 0).to(device)
            new_shape = image.shape
            resized_attr = attr.reshape(attr.shape[:2] + new_shape[2:])

        infidelity_per_batch = infidelity(forward_fn,
                                            perturb_func=perturb_fn,
                                            inputs=image, 
                                            attributions=attr, 
                                            max_examples_per_batch=bsz,
                                            target=target,
                                            n_perturb_samples=39)
        infidelity_list.append(round(infidelity_per_batch[1].item(), 5))
    
    return infidelity_list


def visualize_consistency(images, methods, bboxes, scores, cmap, file_path=None,
                    resize=True, save_fig=True):
    '''Returns a figure of visualized attributes calculated for several slices of one study case
        Args:
            images: list[list[tensor/ImageList]] - list of attributes that should be displayed
            methods: list[str] - list of methods' titles
            bboxes: list[tensors] - list of predicted bounding boxes
            scores: list[tensors] - list of predicted scores
            cmap: matplotlib.colors.Colopmap or other - colormap to visualize attributes
            file_path: str(optional) - is needed when save_fig is True
            resize: boolean - whether to resize a visual representation or not
            save_fig: boolean - saves figure to the image'''
                    
    scale = 2

    columns = len(images[0]) 
    rows = len(methods)

    fig, ax = plt.subplots(rows, columns, figsize=(3*columns, 2*rows))
    
    for i in range(rows):
        attrs_row = images[i]

        for j in range(columns):
            image = attrs_row[j] 
            
            
            if isinstance(image, ImageList):
                image = normalize_image(image.tensors[1, ...])

            if resize:
                image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            ax[i, j].imshow(image, cmap=cmap)
            
            bbox = bboxes[j]
            score = scores[j]
            if len(bbox) != 0:
                for l in range(bbox.shape[0]):
                    box = list(map(int, bbox[l]))
                    width, height = box[2]-box[0],  box[3]-box[1]
                    score_ = str(round(score[l], 3))
                    ax[i, j].add_patch(matplotlib.patches.Rectangle(tuple(box[:2]), width, height, linewidth=1,
                                edgecolor='r', facecolor='none')) 
                    ax[i, j].text(box[0], box[1]-5, score_, fontsize=10, c='red')
            
            #ax[i, j].axis('off')
            if j == 0:
                ax[i, j].set_ylabel(methods[i])
    if save_fig:
        fig.savefig(file_path, dpi=300)
            
    plt.show()    
