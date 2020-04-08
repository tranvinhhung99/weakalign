from __future__ import print_function, division
import torch
import os
from os.path import exists
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric
from image.normalization import NormalizeImageDict, normalize_image
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import warnings
from torchvision.transforms import Normalize
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')

SELECTIONS = [
    'proposed_resnet101',
    'cnngeo_vgg16',
    'cnngeo_resnet101'

]

resizeCNN = GeometricTnf(out_h=240, out_w=240, use_cuda = False) 
def preprocess_image(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)
    
    # Normalize image
    image_var = normalize_image(image_var)
    
    return image_var 


def affTpsTnf(source_image, theta_aff, theta_aff_tps, use_cuda=True):
    tpstnf = GeometricTnf(geometric_model = 'tps', use_cuda=use_cuda)
    sampling_grid = tpstnf(image_batch=source_image,
                           theta_batch=theta_aff_tps,
                           return_sampling_grid=True)[1]
    X = sampling_grid[:,:,:,0].unsqueeze(3)
    Y = sampling_grid[:,:,:,1].unsqueeze(3)
    Xp = X*theta_aff[:,0].unsqueeze(1).unsqueeze(2)+Y*theta_aff[:,1].unsqueeze(1).unsqueeze(2)+theta_aff[:,2].unsqueeze(1).unsqueeze(2)
    Yp = X*theta_aff[:,3].unsqueeze(1).unsqueeze(2)+Y*theta_aff[:,4].unsqueeze(1).unsqueeze(2)+theta_aff[:,5].unsqueeze(1).unsqueeze(2)
    sg = torch.cat((Xp,Yp),3)
    warped_image_batch = F.grid_sample(source_image, sg)

    return warped_image_batch


class AffineModel(object):
    def __init__(self, model_selection='proposed_resnet101', use_cuda=True):
        self.model_selection = model_selection
        self.use_cuda = use_cuda

        self.build_model(model_selection, use_cuda)

        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)


    def build_model(self, model_selection, use_cuda):
        if model_selection=='cnngeo_vgg16':
            model_aff_path = 'trained_models/trained_models/cnngeo_vgg16_affine.pth.tar'
            model_tps_path = 'trained_models/trained_models/cnngeo_vgg16_tps.pth.tar'
            feature_extraction_cnn = 'vgg'
            
        elif model_selection=='cnngeo_resnet101':
            model_aff_path = 'trained_models/trained_models/cnngeo_resnet101_affine.pth.tar'
            model_tps_path = 'trained_models/trained_models/cnngeo_resnet101_tps.pth.tar'   
            feature_extraction_cnn = 'resnet101'
            
        elif model_selection=='proposed_resnet101':
            model_aff_tps_path = 'trained_models/weakalign_resnet101_affine_tps.pth.tar'
            feature_extraction_cnn = 'resnet101'

        self.model = TwoStageCNNGeometric(use_cuda=use_cuda,
                                     return_correlation=False,
                                     feature_extraction_cnn=feature_extraction_cnn)
        
        if model_selection == 'proposed_resnet101':
            self.load_weight_proposed_resnet(model_aff_tps_path)

        

    def load_weight_proposed_resnet(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])


    def load_image(self, img):
        if isinstance(img, str):
            image_path = img
            #img = io.imread(image_path)
            img = np.array(Image.open(image_path))
        img_tensor = preprocess_image(img)
        if self.use_cuda:
            return img_tensor.cuda(), img.shape
        else:
            return img_tensor, img.shape


    def process(self, source_img_path, target_img_path):
        """ Main process of model 
        
        Input: 
            source_img_path: path to image you want to transform or np.array (RGB)
            target_img_path: path to target image you want to transform or np.array (RGB)

        Output:
            affine_image (np.ndarray same shape with target) Affine Transformation result 
            affine_tps_image (np.ndarray same shape with target): Affine TPS transformation result
            
        """
        # Load data
        src_img, src_shape = self.load_image(source_img_path)
        target_img, target_shape = self.load_image(target_img_path)
        
        batch = {
            'source_image': src_img,
            'target_image': target_img
        }


        self.model.eval()

        with torch.no_grad():
            theta_aff,theta_aff_tps = self.model(batch) 
        resizeTgt = GeometricTnf(out_h=target_shape[0], out_w=target_shape[1], use_cuda = self.use_cuda) 
        
        warped_image_aff = self.affTnf(batch['source_image'],theta_aff.view(-1,2,3))
        warped_image_aff_tps = affTpsTnf(batch['source_image'],theta_aff,theta_aff_tps)

        warped_image_aff_np = normalize_image(resizeTgt(warped_image_aff),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        warped_image_aff_tps_np = normalize_image(resizeTgt(warped_image_aff_tps),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

        warped_image_aff_tps_np = warped_image_aff_tps_np.astype(np.uint8)
        warped_image_aff_np = warped_image_aff_np.astype(np.uint8)

        return warped_image_aff_np, warped_image_aff_tps_np


if __name__ == '__main__':
    model = AffineModel()
    sample_data_folder = "datasets/ai_city/sample/"
    target_image_path = sample_data_folder + '000017__id_0_label_1.jpg'
    source_image_path = sample_data_folder + '000168__id_1_label_1.jpg'

    aff, aff_tps = model.process(source_image_path, target_image_path)

