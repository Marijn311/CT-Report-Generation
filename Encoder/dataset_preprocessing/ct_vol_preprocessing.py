import torch
import numpy as np
from scipy.ndimage import zoom
from config import *

"""
This file contains various small functions which can be used for CT volume preprocessing.
The main function called preprocess_ct_volume is used in the __getitem__ function of the dataset class.
Therefore all the functions in this file are used to preprocess a single CT volume.
The preprocess_ct_volume function, which can be found at the bottom, consists of a sequence these smaller functions.
Provided functions include: 
    -padding
    -cropping
    -reshaping to 1 or 3 channels
    -data augmentation (flips and rotations)
    -converting tensors to torch
    -normalizing pixel values
    -clipping (HU) pixel values
    -centering pixel values on the ImageNet mean
    -rescaling masked images to better fill available space

Most of the functions are taken from: https://github.com/rachellea/ct-net-models/blob/master/load_dataset/utils.py
"""

###########################################################################################################
# Pixel Values (normalizing, clipping, centering, converting to torch) #-----------------------------------
###########################################################################################################
def normalize(ctvol, lower_bound, upper_bound):
    """Clip and then normalize pixel values"""
    ctvol = torch.clamp(ctvol, lower_bound, upper_bound)
    ctvol = (ctvol - lower_bound) / (upper_bound - lower_bound)
    return ctvol

def torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds, center_on_imagenet_mean):
    """Normalize using specified pixel_bounds and then center on the ImageNet
    mean."""
    # Cast to torch Tensor for speed
    ctvol = torch.from_numpy(ctvol).type(torch.float)
    
    # Clip Hounsfield units and normalize pixel values
    ctvol = normalize(ctvol, pixel_bounds[0], pixel_bounds[1])
    
    # If requested: Center on the ImageNet mean (for when you are using an ImageNet weights)
    if center_on_imagenet_mean == True:
        ctvol = ctvol - 0.449
    ctvol = ctvol - 0.449
    return ctvol


###########################################################################################################
# Padding (both in the slice plane and the axial plane) #--------------------------------------------------
###########################################################################################################
def pad_slices(ctvol, max_slices, padding_value):
    """For <ctvol> of shape (slices, side, side) pad the slices to shape
    max_slices for output of shape (max_slices, side, side)"""
    padding_needed = max_slices - ctvol.shape[0]
    assert (padding_needed >= 0), 'Image slices exceed max_slices by'+str(-1*padding_needed)
    if padding_needed > 0:
        before_padding = int(padding_needed/2.0)
        after_padding = padding_needed - before_padding
        ctvol = np.pad(ctvol, pad_width = ((before_padding, after_padding), (0,0), (0,0)),
                     mode = 'constant', constant_values = padding_value)
        assert ctvol.shape[0]==max_slices, "Something went wrong with the padding of the slices. The number of slices is not equal to max_slices."
    return ctvol

def pad_sides(ctvol, max_side_length, padding_value): 
    """For <ctvol> of shape (slices, side, side) pad the sides to shape
    max_side_length for output of shape (slices, max_side_length,
    max_side_length)"""
    needed_padding = 0
    for side in [1,2]:
        padding_needed = max_side_length - ctvol.shape[side]
        if padding_needed > 0:
            before_padding = int(padding_needed/2.0)
            after_padding = padding_needed - before_padding
            if side == 1:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (before_padding, after_padding), (0,0)),
                         mode = 'constant', constant_values = padding_value)
                needed_padding += 1
            elif side == 2:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (0,0), (before_padding, after_padding)),
                         mode = 'constant', constant_values = padding_value)
                needed_padding += 1
    if needed_padding == 2:
        assert ctvol.shape[1]==ctvol.shape[2]==max_side_length, "The side lengths are not equal"
    return ctvol

def pad_volume(ctvol, max_slices, max_side_length, padding_value):
    """Pad <ctvol> to a minimum size of
    [max_slices, max_side_length, max_side_length]"""
    if ctvol.shape[0] < max_slices:
        ctvol = pad_slices(ctvol, max_slices, padding_value)
    if ctvol.shape[1] < max_side_length:
        ctvol = pad_sides(ctvol, max_side_length, padding_value)
    return ctvol


###########################################################################################################
# Reshaping to 3 Channels #-------------------------------------------------------------------------------
###########################################################################################################
def make_chunks(ctvol):
    """Given a numpy array <ctvol> with shape [slices, square, square]
    reshape to 'RGB' [max_slices/3, 3, square, square]"""
    return np.reshape(ctvol, newshape=[int(ctvol.shape[0]/3), 3, ctvol.shape[1], ctvol.shape[2]])

def reshape_3_channels(ctvol):
    """Reshape grayscale <ctvol> to a 3-channel image. Making chunks of 3 slices to pass to an RGB model"""
    if ctvol.shape[0]%3 == 0:
        ctvol = make_chunks(ctvol)
    else:
        if (ctvol.shape[0]-1)%3 == 0:
            ctvol = make_chunks(ctvol[:-1,:,:])
        elif (ctvol.shape[0]-2)%3 == 0:
            ctvol = make_chunks(ctvol[:-2,:,:])
    return ctvol


###########################################################################################################
# Cropping (both in the slice plane and the axial plane) and Data Augmentation (flips and rotations) #-----
###########################################################################################################
def crop_specified_axis(ctvol, max_dim, axis): 
    """Crop 3D volume <ctvol> to <max_dim> along <axis>"""
    dim = ctvol.shape[axis]
    if dim > max_dim:
        amount_to_crop = dim - max_dim
        part_one = int(amount_to_crop/2.0)
        part_two = dim - (amount_to_crop - part_one)
        if axis == 0:
            return ctvol[part_one:part_two, :, :]
        elif axis == 1:
            return ctvol[:, part_one:part_two, :]
        elif axis == 2:
            return ctvol[:, :, part_one:part_two]
    else:
        return ctvol

def crop_3d(ctvol, max_slices, max_side_length):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length]"""
    ctvol = crop_specified_axis(ctvol, max_slices, 0)
    ctvol = crop_specified_axis(ctvol, max_side_length, 1)
    ctvol = crop_specified_axis(ctvol, max_side_length, 2)
    return ctvol

def crop_3d_augment(ctvol, max_slices, max_side_length):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length] with randomness in the centering and random
    flips or rotations"""
    # Introduce different amount of random padding on each side such that the center and thus the crop will be slightly random
    ctvol = rand_pad(ctvol)
    
    # Obtain the center crop
    ctvol = crop_3d(ctvol, max_slices, max_side_length)
    
    # Flip and rotate
    ctvol = rand_flip(ctvol)
    ctvol = rand_rotate(ctvol)
    
    # Make contiguous array to avoid Pytorch error
    return np.ascontiguousarray(ctvol)

def rand_pad(ctvol):
    """Introduce random padding between 0 and 15 pixels on each of the 6 sides
    of the <ctvol>"""
    randpad = np.random.randint(low=0,high=15,size=(6))
    ctvol = np.pad(ctvol, pad_width = ((randpad[0],randpad[1]), (randpad[2],randpad[3]), (randpad[4], randpad[5])),
                         mode = 'constant', constant_values = np.amin(ctvol))
    return ctvol
    
def rand_flip(ctvol):
    """Flip <ctvol> along a random axis with 50% probability"""
    if np.random.randint(low=0,high=100) < 50:
        chosen_axis = np.random.randint(low=0,high=3)
        ctvol =  np.flip(ctvol, axis=chosen_axis)
    return ctvol

def rand_rotate(ctvol):
    """Rotate <ctvol> the slices some random amount of 90 degrees with 50% probability"""
    if np.random.randint(low=0,high=100) < 50:
        chosen_k = np.random.randint(low=0,high=4)
        ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2))
    return ctvol


###########################################################################################################
# Rescaling Masked Images (to better fill available image size) #------------------------------------------
###########################################################################################################
def rescale_image(ctvol, desired_shape, padding_value):
    """Find the smallest subarray of <ctvol> containing the masked object.
    Rescale the subarray, maintaining aspect ratios, until it optimally fits the available image size.
    If the inputted image is not masked, the entire image will be rescaled to fit the available image size."""
    
    # Find the indices of the non-background subarray.
    roi_indices = np.where(ctvol != padding_value)

    # Find the minimum and maximum indices along each axis
    min_indices = np.min(roi_indices, axis=1)
    max_indices = np.max(roi_indices, axis=1)

    # Slice the original image to extract the smallest subarray containing the object
    ctvol_roi = ctvol[  min_indices[0]:max_indices[0]+1,
                        min_indices[1]:max_indices[1]+1,
                        min_indices[2]:max_indices[2]+1]

    # Calculate the scaling ratio needed for each axis to upscale the subarray to the available space
    scaling_ratios = [desired_shape[i] / ctvol_roi.shape[i] for i in range(3)]
    
    # Take the smallest scaling ratio
    # If the roi has to be downscaled to fit the available space then the ratio will be below 1 
    # The smallest ratio below 1 will cause the most extreme downscaling, meaning that the entire roi will fit in the available space
    # Hence taking the smallest ratio works for both upscaling and downscaling
    smallest_ratio = min(scaling_ratios)
  
    # Rescale the image_roi in all dimension with the smallest scaling ratio, this keeps the aspect ratio
    rescaled_roi = zoom(ctvol_roi, zoom=smallest_ratio, order=1, cval=padding_value)
    
    # Make sure that none of the dimensions of the rescaled_roi are larger than the available space 
    assert all([rescaled_roi.shape[i] <= desired_shape[i] for i in range(3)]), "The rescaled_roi is larger than the available space"
    
    
    # Print the smallest_ratio and the shape of the rescaled_roi and the original image_roi
    if SHOW_DATA_EXAMPLES==True:
        print(f"Original (istropic spacing) shape: {ctvol.shape}")
        print(f"Original roi shape: {ctvol_roi.shape}")
        print(f"Rescaled roi shape: {rescaled_roi.shape}")
        print(f"Rescaling ratio: {smallest_ratio}")
        
    return rescaled_roi, smallest_ratio

###########################################################################################################
# Main function: Dataset Preprocessing Sequences for the __getitem__ #-------------------------------------
###########################################################################################################
def preprocess_ct_volume(ctvol, pixel_bounds, data_augment, num_channels,
                                  max_slices, max_side_length, center_on_imagenet_mean, padding_value):
    """This function processes only 1 images a the time, it is called in the __getitem__ function of the dataset class.
   
    <ctvol> is a single numpy array of shape [slices, side, side].
    <pixel_bounds> is a list of ints e.g. [-1000,1300] Hounsfield units. Used for pixel value clipping and normalization.
    <data_augment> is a Boolean. If set to true there is a 50% chance that the image is flipped along a random axis, 
        and there is an independent 50% chance that the image will be rotated some random amount of 90 degrees. 
    <num_channels> is an int, e.g. 3 to reshape the grayscale volume into chunks of 3 slices.
    <max_slices> is an int, the maximum number of slices.
    <max_side_length> is an int, the maximum side length of a slice.
    <center_on_imagenet_mean> is a Boolean. If set to true the pixel values are centered on the ImageNet mean.
    """
    
    assert ctvol.shape[1] == ctvol.shape[2], "The images should have shape [slices, sides, sides]"
    
    # Rescale the image to fit the volume as best as possible. 
    ctvol, _ = rescale_image(ctvol, desired_shape=(max_slices, max_side_length, max_side_length), padding_value=padding_value)

    # Pad the image to minimum size [max_slices, max_side_length, max_side_length]
    ctvol = pad_volume(ctvol, max_slices, max_side_length, padding_value)
    
    # Cropping, and data augmentation if indicated
    if data_augment == True:
        ctvol = crop_3d_augment(ctvol, max_slices, max_side_length)
    else:
        ctvol = crop_3d(ctvol, max_slices, max_side_length)
    
    # Reshape to 3 channels if indicated. This entails making chunks of 3 slices. [slices, square, square] -> [max_slices/3, 3, square, square]
    assert num_channels == 3 or num_channels == 1, "The number of channels should be 1 or 3"
    if num_channels == 3:
        ctvol = reshape_3_channels(ctvol)
    if num_channels == 1:
        ctvol = np.expand_dims(ctvol, axis=0)
    
    # Cast to torch tensor, clip and normalize pixel values, and center on the ImageNet mean, if requested
    output = torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds, center_on_imagenet_mean)

    return output
