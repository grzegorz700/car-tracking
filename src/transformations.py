import cv2
import numpy as np


class MorphologicalTransformer:
    """Adjust and apply morphological transformations"""

    def __init__(self, window_name, max_close_kernel=35):
        self.window_name = window_name
        self.max_close_kernel = max_close_kernel
        self.__track_bars_created = False

        # Prepare gaussian kernels:
        gauss_kernels = []
        for i in range(1, max_close_kernel + 1):
            kernel_gauss_1d = cv2.getGaussianKernel(i, -1)
            kernel_gauss_2d = np.matmul(kernel_gauss_1d, kernel_gauss_1d.transpose())
            gauss_kernels.append(kernel_gauss_2d)
        self.__gauss_kernels = gauss_kernels

        # Prepare simple kernels (array with ones)
        simple_kernels = []
        for size_of_kernel in range(1, max_close_kernel + 1):
            kernel = np.ones((size_of_kernel, size_of_kernel), np.uint8)
            simple_kernels.append(kernel)
        self.__simple_kernels = simple_kernels

    def create_trackbars(self):
        def nothing(_): pass
        cv2.createTrackbar('Erode', self.window_name, 2, 20, nothing)
        cv2.createTrackbar('Dilate', self.window_name, 2, 20, nothing)
        cv2.createTrackbar('Blur', self.window_name, 12, 20, nothing)
        cv2.createTrackbar('Kernel_OPEN', self.window_name, 0, 20, nothing)
        cv2.createTrackbar('Kernel_CLOSE', self.window_name, 25, self.max_close_kernel, nothing)
        cv2.createTrackbar('Shadows', self.window_name, 0, 200, nothing)

    def setup_shadows_settings(self, background_subtractor):
        shadows = cv2.getTrackbarPos('Shadows', self.window_name)
        if shadows == 0:
            background_subtractor.setDetectShadows(False)
        else:
            background_subtractor.setDetectShadows(True)
            background_subtractor.setShadowThreshold(shadows)

    def apply_erode(self, mask_org):
        size_of_kernel_erode = cv2.getTrackbarPos('Erode', self.window_name)
        mask_out = mask_org
        if size_of_kernel_erode > 0:
            kernel_erode = self.__simple_kernels[size_of_kernel_erode-1]
            mask_out = cv2.erode(mask_org, kernel_erode, iterations=1)
        return mask_out

    def apply_dilate(self, mask_org):
        size_of_kernel_dilate = cv2.getTrackbarPos('Dilate', self.window_name)
        mask_out = mask_org
        if size_of_kernel_dilate > 0:
            kernel_dilate = self.__simple_kernels[size_of_kernel_dilate-1]
            mask_out = cv2.dilate(mask_org, kernel_dilate, iterations=1)
        return mask_out

    def apply_morph_open(self, mask_org):
        size_of_kernel_open = cv2.getTrackbarPos('Kernel_OPEN', self.window_name)
        mask_out = mask_org
        if size_of_kernel_open > 0:
            kernel_open = self.__gauss_kernels[size_of_kernel_open - 1]
            mask_out = cv2.morphologyEx(mask_org, cv2.MORPH_OPEN, kernel_open)
        return mask_out

    def apply_morph_close(self, mask_org):
        size_of_kernel_close = cv2.getTrackbarPos('Kernel_CLOSE', self.window_name)
        mask_out = mask_org
        if size_of_kernel_close > 0:
            kernel_close = self.__gauss_kernels[size_of_kernel_close - 1]
            mask_out = cv2.morphologyEx(mask_org, cv2.MORPH_CLOSE, kernel_close)
        return mask_out

    def apply_blur(self, frame):
        size_of_kernel_blur = cv2.getTrackbarPos('Blur', self.window_name)
        result_frame = frame
        if size_of_kernel_blur > 0:
            size_real = (size_of_kernel_blur - 1) * 2 + 1
            result_frame = cv2.GaussianBlur(frame, (size_real, size_real), 0)
        return result_frame
