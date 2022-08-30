from captum.attr import GradientShap, LayerGradientShap, LayerActivation, Lime, Occlusion
from captum.attr._utils.visualization import _normalize_image_attr
from captum.attr._core.lime import get_exp_kernel_similarity_function
from model.data.utils import normalize_image, draw_pred_bboxes
from matplotlib.colors import LinearSegmentedColormap
from maskrcnn.structures.image_list import ImageList
from captum.attr import visualization as viz
from skimage.segmentation import quickshift
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import torch
import cv2


class Explainer:   
    def _get_explainer(self, method_title, forward_fn, **kwargs):
        if method_title == 'GradientShap':
            return GradientShapExplainer(forward_fn)
        elif method_title == 'LayerGradientShap':
            return LayerGradientShapExplainer(forward_fn, kwargs['model'])
        elif method_title == 'LayerActivation':
            return LayerActivationExplainer(forward_fn, kwargs['model'])
        elif method_title == 'Occlusion':
            return OcclusionExplainer(forward_fn)
        elif method_title == 'LimeExplainer':
            return LimeExplainer(forward_fn, kwargs['kernel_width'], kwargs['interpretable_model'])
                        
    def get_explanations(self, method_title, inputs, forward_fn, **kwargs):
        attributes = []
        if not isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
        for image in tqdm(inputs):
            method = self._get_explainer(method_title, forward_fn, **kwargs)
            attrs = method.get_attributes(image, kwargs['target'])
            attributes.append(attrs)
        return attributes
    
    def visualize_explanations(self, attributes, bboxes, scores, cmap=None):
        if cmap == None:
            cmap = LinearSegmentedColormap.from_list('blue', 
                                                 [(0, '#0000ff'),
                                                  (0.25, '#ffff00'),
                                                  (1, '#ff0000')], N=256) 
        draw_pred_bboxes(attributes, bboxes, scores, cmap)
        

class GradientShapExplainer:
    def __init__(self, forward_fn):
        self.forward_fn = forward_fn
        self.method = GradientShap(self.forward_fn)
    
    def get_attributes(self, image, target):
        if isinstance(image, ImageList):
            image = image.tensors.data
        image = Variable(image, requires_grad=True)
        baselines = Variable(torch.zeros_like(image), requires_grad = True)            
        attributes = self.method.attribute(image, 
                                          baselines, 
                                          target=target, 
                                          n_samples=1, 
                                          stdevs=0.0001)[1, ...]
        attrs_np = attributes.squeeze().cpu().permute(1,2,0).detach().numpy()
        norm_attributes = _normalize_image_attr(attrs_np, sign='absolute_value')                   

        return norm_attributes


class LayerGradientShapExplainer:
    def __init__(self, forward_fn, model, layer=None):
        self.forward_fn = forward_fn
        if layer == None:
            layer = model.backbone.transition2.conv
        self.method = LayerGradientShap(self.forward_fn, layer)
    
    def get_attributes(self, image, target):
        if isinstance(image, ImageList):
            image = image.tensors.data
        image = Variable(image, requires_grad=True)
        baselines = Variable(torch.zeros_like(image), requires_grad = True)            
        attributes = self.method.attribute(image, 
                                          baselines, 
                                          target=target, 
                                          n_samples=1, 
                                          stdevs=0.0001)[1, ...]
        attrs_np = attributes.squeeze().cpu().permute(1,2,0).detach().numpy()
        norm_attributes = _normalize_image_attr(attrs_np, sign='absolute_value')
        new_size = image.shape[-2:][::-1]
        norm_attributes = cv2.resize(norm_attributes, dsize=new_size, interpolation=cv2.INTER_NEAREST)
        norm_attributes = cv2.bilateralFilter(norm_attributes, 9, 75, 75)
        
        return norm_attributes

class LayerActivationExplainer:
    def __init__(self, forward_fn, model, layer=None):
        self.forward_fn = forward_fn
        if layer == None:
            layer = model.backbone.transition2.conv
        self.method = LayerActivation(self.forward_fn, layer)
    
    def get_attributes(self, image, target):
        if isinstance(image, ImageList):
            image = image.tensors.data
        image = Variable(image, requires_grad=True)
        attributes = self.method.attribute(image)[1, ...]
        attrs_np = attributes.squeeze().cpu().permute(1,2,0).detach().numpy()
        norm_attributes = _normalize_image_attr(attrs_np, sign='absolute_value')
        new_size = image.shape[-2:][::-1]
        norm_attributes = cv2.resize(norm_attributes, dsize=new_size, interpolation=cv2.INTER_NEAREST)
        norm_attributes = cv2.bilateralFilter(norm_attributes, 9, 75, 75)

        return norm_attributes
    
    
class OcclusionExplainer:
    def __init__(self, forward_fn):
        self.forward_fn = forward_fn
        self.method = Occlusion(self.forward_fn)
    
    def get_attributes(self, image, target):
        if isinstance(image, ImageList):
            image = image.tensors.data
        image = Variable(image, requires_grad=True)
        attributes = self.method.attribute(image, strides = ((1, 4, 4)), sliding_window_shapes=(1, 7, 7),
                                           target=target, baselines=0)
        attrs_np = attributes.squeeze().cpu().permute(1,2,0).detach().numpy()
        norm_attributes = _normalize_image_attr(attrs_np, sign='all')
        
        
        return norm_attributes
    
    
class LimeExplainer:
    def __init__(self, forward_fn, kernel_width, interpretable_model):
        self.forward_fn = forward_fn  
        self.kernel_width = kernel_width
        self.interpretable_model = interpretable_model
        self.distance = get_exp_kernel_similarity_function('cosine', self.kernel_width)
        self.method = Lime(self.forward_fn, 
                          interpretable_model=self.interpretable_model,  
                          similarity_func=self.distance)

    def get_attributes(self, image, target):
        if isinstance(image, ImageList):
            image = image.tensors
        mask = self.create_seg_mask(image)

        attributes = self.method.attribute(image,
                              target=target,
                              feature_mask=mask,
                              n_samples=351, 
                              perturbations_per_eval=1,
                              baselines=0,
                              show_progress=False)

        attrs_np = np.transpose(attributes[1, ...].cpu().detach().numpy(), (1,2,0))
        norm_attributes = viz._normalize_image_attr(attrs_np, sign='all')

        return norm_attributes

    def create_seg_mask(self, image):
        norm_image = normalize_image(image[1, ...]) 
        segmented_img = quickshift(norm_image, kernel_size=8)
        segmented_img_exp = np.tile(np.expand_dims(segmented_img, axis=0), (3, 1, 1))

        mask_exp = np.expand_dims(segmented_img_exp, axis=0)
        mask_exp = np.tile(mask_exp, (3, 1, 1, 1))
        mask_tensor = torch.as_tensor(mask_exp, dtype=torch.long, 
                                      device=torch.device('cuda:0'))

        return mask_tensor