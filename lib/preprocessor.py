import numpy as np
import cv2 as cv


class Preprocessor:

    def __init__(self,
                 video_size:            tuple[int, int],
                 kernel_size:           float,
                 contrast_quant:        float,
                 reflection_th:         int = 252,
                 dilation_kernel_size:  float = 0.05,
                 mask_r:                float = None,
                 mask_pos:              tuple = None):
        """
        Initialize the Preprocessor.

        :param video_size: video size as [width, height]
        :param kernel_size: relative box blur filter kernel size in each dimension for highpass filtering
        :param contrast_quant: Inclusion quantile for automatic contrast calculation. 
                E.g. 0.95 excludes highest/lowest 5% of image values. Useful against outliers.
        :param reflection_th: reflection detection threshold as int, should be <256
        :param dilation_kernel_size: kernel size for dilation of the reflection mask in relative image units (0.-1.)
        :param mask_r: circular mask radius relative to image (values in [0...1]) 
        :param mask_pos: circular mask center position as tuple [x, y]
        """
        self.kernel_size = kernel_size
        self.video_size = video_size
        self.contrast_quant = contrast_quant
        self.reflection_threshold = reflection_th
        self.dilation_kernel_size = int(dilation_kernel_size * min(self.video_size))  # convert to absolute pixels
        self.mask_r = mask_r
        self.mask_pos = mask_pos
       
        # initialize circular mask
        if self.mask_r is not None and self.mask_pos is not None:
            Y, X = np.mgrid[0:self.video_size[1], 0:self.video_size[0]]
            R2 = (X - self.mask_pos[0]*self.video_size[0])**2 + (Y - self.mask_pos[1]*self.video_size[1])**2
            self.mask = R2 > (self.mask_r*self.video_size[0])**2
        else:
            self.mask = None

    def process(self, img_in: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame.
        Includes high pass filtering, reflection removal and contrast enhancement.

        :param img_in: input greyscale image
        :return: processed image (type np.uint8 and shape as input)
        """
        # image padding
        k2 = int((self.kernel_size * self.video_size[0]) // 2) + 1
        img_pad = np.pad(img_in, ((k2, k2), (k2, k2)), mode="edge")
        
        # high pass filter 
        img_lp = cv.blur(img_pad, (round(self.kernel_size*self.video_size[0]),
                                  round(self.video_size[0]*self.kernel_size)))
        img_lp = img_lp[k2:-k2, k2:-k2]
        diff = img_in - img_lp
        
        # reflection removal with thresholding and dilation
        clipping = img_in > self.reflection_threshold
        dkernel = [self.dilation_kernel_size, self.dilation_kernel_size]
        clipping = cv.dilate(clipping.astype(np.uint8), kernel=cv.getStructuringElement(cv.MORPH_RECT, dkernel)) > 0
        diff[clipping] = 0
        
        # mask to circular region
        if self.mask is not None:
            diff[self.mask] = 0
       
        # calculate contrast factor
        ptp1 = np.ptp(diff, axis=1)
        ptp = 255 / (1 + np.quantile(ptp1, self.contrast_quant))
        gain = max(1, min(30, ptp))
        
        # apply contrast and convert image back to uint8
        img_out = gain*diff + 127.0
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out.astype(np.uint8)

        return img_out

