# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 01:12:21 2022

@author: Dell
"""
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt




# First we need to create a function that add a periodic noise
def add_periodic_noise(img, A=100, u0=90, v0=50, freq = 100):
    #A: amplitude - int.
    #u0: angle 
    #v0: angle
    shape = img.shape
    
    noise = np.zeros((shape[0], shape[1]), dtype='float32')
    x, y = np.meshgrid(range(0, shape[1]), range(0, shape[0]))

    noise += A * np.sin((x * u0 + y * v0) / freq)

    if len(img.shape) > 2:
        noise = noise.reshape(shape[0], shape[1], 1)

    return img + noise



def shifted_dft(img):
    
    # Transform image from spatial domain to frequency domain by dft Then shift to zero
    # Inputs: img
    # output: shifted_dft (a numpy array containing the transformed image with the same size
    # but with 2 channels (real & complex).)
    img_dft = np.fft.fft2(img)
    shifted_dft = np.fft.fftshift(img_dft)

    return shifted_dft


def dft_magnitude(shifted_dft):
    
    # Compute the magnitude spectrum of an image in the frequency domain
    # Inputs: shifted_dft
    # Output: magnitude_spectrum ==> a numpy array of shape (H, W) containing intensity values.
    magnitude_spectrum = 20* np.log(np.abs(shifted_dft)+1)
    return magnitude_spectrum


def inverse_shifted_dft(shifted_dft):
    
    # Calculates the inverse Discrete Fourier Transform of an image after shifting the zero frequency component
    # Inputs: shifted_dft
    #Outputs : img
    img_dft = np.fft.ifftshift(shifted_dft)
    img = np.fft.ifft2(img_dft)
    img = np.abs(img)

    return img


def get_dft_indx(magnitude_spectrum):
    # This function to return the index of the dft image
    # Input: dft_magnitude_value
    # Output: indx: tuple of index ((x,y), (x,y)).
    
    magnitude_spectrum = copy.deepcopy(magnitude_spectrum)
    dft_indexs = ()

    # Get the 3 max pixel values
    while len(dft_indexs) < 3:
        ind = np.unravel_index(magnitude_spectrum.argmax(), magnitude_spectrum.shape)

        # make the max value of pixel to be zero so not be selected again.
        magnitude_spectrum[ind[0], ind[1]] = 0

        # Add the pixel index
        dft_indexs = dft_indexs + (ind,)
    return dft_indexs



def notch_filter(image, offest = 0):
   
    # This function to remove periodic noise by making the dft row and column to be zeros
    # Input : an image with periodic noise.
    # Output: an image without periodic noise.
    def set_dft_to_zero(dft_indx, dft_value, offset = 0):
        """
        
        """
        # This function to make the row and column of dft to be zero
        # Input: dft_indx, dft_value, offset
        #
        # Output: dft_value
        
        dft_indx = list(dft_indx)
        dft_indx1 = []
        if offset:

            for i in range(1, offset + 1):
                for dft_ind in dft_indx:
                    dft_indx1.append((dft_ind[0] + i, dft_ind[1]))
                    dft_indx1.append((dft_ind[0] - i, dft_ind[1]))

                    dft_indx1.append((dft_ind[0], dft_ind[1] + i))
                    dft_indx1.append((dft_ind[0], dft_ind[1] - i))

        dft_indx.extend(dft_indx1)
        for dft_ind in dft_indx:
            dft_value[dft_ind[0], :] = 0
            dft_value[:, dft_ind[1]] = 0
        return dft_value

    # Get the DFT of the image
    dft_value = shifted_dft(image)

    # Get the magnitude of the DFT
    dft_magnitude_value = dft_magnitude(dft_value)

    # Get the index of the DFT
    dft_indx = get_dft_indx(dft_magnitude_value)[1:]

    # Set the dft to zero
    filtered_dft = set_dft_to_zero(dft_indx, dft_value, offest)

    # get the image back
    img_back = inverse_shifted_dft(filtered_dft)

    return img_back


#%%
img = cv2.imread("What Causes Eye Floaters.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



img_with_periodic_noise = add_periodic_noise(img_gray, 200, 50, 70, 80)

plt.imshow(img_with_periodic_noise, cmap = 'gray')
plt.show()
filtered_image = notch_filter(img_with_periodic_noise, 10)

plt.imshow(filtered_image, cmap = 'gray')

plt.show()

