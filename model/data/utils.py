from maskrcnn.structures.image_list import ImageList
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import cv2


def normalize_image(image, max_value=255, dtype=np.uint8):
    '''Converts tensor image representation to numpy array notmalized to range from 0 to max_value
        Args:
            image: tensor - input image to normalize
            max_value: int - maximum value of a range which is 0 or 255
            dtype: data-type - the desired data-type for the array'''
    
    if torch.is_tensor(image):
        image = image.squeeze().cpu().detach().numpy()
    image -= image.min()
    image /= image.max()
    image *= max_value
    if image.shape[0] == 3:
        image = np.transpose(image.astype(dtype), (1, 2, 0))
        
    return image


def draw_pred_bboxes(images, bboxes, scores, cmap='gray', resize=True, save_fig=False, file_path=None):
    
    '''Draw bounding boxes in the given list of images representing slices of nifti file
        Args:
            images: list(ndarray) - list of np arrays with (H, W, C) shape
            bboxes: list(ndarray) - list of bounding boxes with (N, 4) shape
    '''
    
    scale = 2 
    if isinstance(images[0], ImageList):
        images = [normalize_image(img.tensors[1, :, :, :]) for img in images]
        
    n = len(images)
    columns = 2
    rows = 0
    k = 0
    if n % 2 == 0:
        rows = int(n // 2)
    else:
        rows = int((n + 1) // 2)
    fig, ax = plt.subplots(rows, columns, figsize=(columns*5, rows*3))
    
    for i in range(rows):
        for j in range(2):
            if k == n:
                image = np.zeros(images[k-1].shape)
            else:
                norm_image = images[k] 
                if resize:
                    norm_image = cv2.resize(norm_image, None, None, fx=scale, fy=scale,
                                            interpolation=cv2.INTER_LINEAR)
                
                ax[i, j].imshow(norm_image, cmap=cmap)

                if len(bboxes[k]) != 0:
                    for l in range(bboxes[k].shape[0]):
                        bbox = list(map(int, bboxes[k][l]))
                        width, height = bbox[2]-bbox[0],  bbox[3]-bbox[1]
                        score = str(round(scores[k][l], 3))
                        ax[i, j].add_patch(matplotlib.patches.Rectangle(tuple(bbox[:2]), width, height, 
                                                                        linewidth=1, edgecolor='r',
                                                                        facecolor='none')) 
                        ax[i, j].text(bbox[0], bbox[1]-5, score, fontsize=10, c='red')
                ax[i, j].axis('off')
                k += 1
    if save_fig:
        fig.savefig(file_path, dpi=300)
            
    plt.show()    