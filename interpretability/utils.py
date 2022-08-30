from model.data.utils import normalize_image
from matplotlib.colors import LinearSegmentedColormap
from maskrcnn.structures.image_list import ImageList
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import matplotlib



def vis_multiple_methods(images,methods, bboxes, scores, show_method_title=True, row_col_num=(4, 4), 
                         cmap='gray', resize=False, save_fig=True, image_title=None):
    
    '''Draw bounding boxes in the given list of images representing slices of nifti file
        Args:
            images: list[list[tensor/ImageList]] - list of attributes that should be displayed
            methods: list[str] - list of methods' titles
            bboxes: list[tensors] - list of predicted bounding boxes
            scores: list[tensors] - list of predicted scores
            show_method_title: boolean - enables display of methods titles
            row_col_num: tuple(int) - sets number of rows and columns in a figure
            cmap: matplotlib.colors.Colopmap or other - colormap to visualize attributes
            image_title: str(optional) - is needed when save_fig is True
            resize: boolean - whether to resize a visual representation or not
            save_fig: boolean - saves figure to the image
    '''
    
    scale = 2 
    rows, columns = row_col_num 
     
    fig, ax = plt.subplots(rows, columns, figsize=(3*columns, 2*rows))
    
    for i in range(rows):
        attrs_row = images[i]
        bbox = bboxes[i]
        score = scores[i]
        for j in range(columns):
            image = attrs_row[j] 
            
            if isinstance(image, ImageList):
                image = normalize_image(image.tensors[1, :, :, :])

            if resize:
                image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            ax[i, j].imshow(image, cmap=cmap)

            if len(bbox) != 0:
                for l in range(bbox.shape[0]):
                    box = list(map(int, bbox[l]))
                    width, height = box[2]-box[0],  box[3]-box[1]
                    score_ = str(round(score[l], 3))
                    ax[i, j].add_patch(matplotlib.patches.Rectangle(tuple(box[:2]), width, height, linewidth=1,
                                edgecolor='r', facecolor='none')) 
                    ax[i, j].text(box[0], box[1]-5, score_, fontsize=10, c='red')
            if i == 0 and show_method_title:
                ax[i, j].set_title(f'{methods[j]}')
            ax[i, j].axis('off')
            #k += 1
    if save_fig:
        fig.savefig(image_title, dpi=300)
            
    plt.show()    