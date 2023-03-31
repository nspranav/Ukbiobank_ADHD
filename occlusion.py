import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import nibabel as nib
from  scipy.ndimage import zoom


def area_occlusion(model, image_tensor, area_masks, target_class=None, occlusion_value=0, apply_softmax=True, cuda=False, verbose=False, taskmode='clx'):
    
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor    
    
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))[0]
    
    if apply_softmax:
        output = F.softmax(output)
    
    if taskmode == 'reg':
        unoccluded_prob = output.data
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data.cpu().numpy()[0]    
        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
        
        if target_class is None:
            target_class = output_class
        unoccluded_prob = output.data[0, target_class]
    
    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()
    
    for area_mask in area_masks:
        area_mask = torch.FloatTensor(area_mask)
        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)
        
        output = model(Variable(image_tensor_occluded[None], requires_grad=False))[0]
        if apply_softmax:
            output = F.softmax(output)
            
        if taskmode == 'reg':
            occluded_prob = output.data
        elif taskmode == 'clx':
            occluded_prob = output.data[0, target_class]
        
        ins = area_mask.view(image_tensor.shape) == 1
        ins = ins.squeeze()
        relevance_map[ins] = (unoccluded_prob - occluded_prob)
    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map

def get_brain_area_masks(data_size):
    
    brain_map = load_nifti('./aal.nii.gz')
    brain_areas = np.unique(brain_map)[1:]  # omit background
    area_masks = []
    for area in brain_areas:
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = resize_image(area_mask, data_size, interpolation=0)
        area_masks.append(area_mask)
    area_names = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
    merged_area_names = [name[:-2] for name in area_names[:108:2]] + area_names[108:]
    
    return area_masks, area_names, merged_area_names



def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())
    
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)
    return struct_arr

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return zoom(img, zoom_factors, order=interpolation)