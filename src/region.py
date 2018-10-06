import numbers

import cv2
import numpy as np


min_length_of_region = 10


class Region:
    def __init__(self, x, y, w, h):
        assert isinstance(x, numbers.Number) \
            and isinstance(y, numbers.Number) \
            and isinstance(w, numbers.Number) \
            and isinstance(h, numbers.Number)
        if w < 0 and h < 0:
            self.was_reversed = True
            if w < 0:
                w = -w
                x -= w
            if h < 0:
                h = -h
                y -= h
        else:
            self.was_reversed = False

        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h

    @staticmethod
    def from_matrix(matrix):
        return Region(matrix[0], matrix[1], matrix[2], matrix[3])

    def is_overlapping(self, other_point):
        if self.x > other_point.x2 or other_point.x > self.x2:
            return False
        if self.y > other_point.y2 or other_point.y > self.y2:
            return False
        return True

    def get_own_subregion(self, predicted_region):
        assert isinstance(predicted_region, Region)

        n_x = max(self.x, predicted_region.x)
        n_y = max(self.y, predicted_region.y)
        n_x2 = min(self.x2, predicted_region.x2)
        n_y2 = min(self.y2, predicted_region.y2)
        n_w = n_x2 - n_x
        n_h = n_y2 - n_y

        result = Region(n_x, n_y, n_w, n_h)
        return result

    @staticmethod
    def merge_regions(regions):
        n_x = min([region.x for region in regions])
        n_x2 = max([region.x2 for region in regions])
        n_y = min([region.y for region in regions])
        n_y2 = max([region.y2 for region in regions])
        n_w = n_x2-n_x
        n_h = n_y2-n_y
        return Region(n_x, n_y, n_w, n_h)

    def get_matrix(self):
        return [self.x, self.y, self.w, self.h]

    def __str__(self):
        return "x={x:.0f},y={y:.0f},w={w:.0f},h={h:.0f}".format(x=self.x, y=self.y, w=self.w, h=self.h)

    def get_area(self):
        return self.w * self.h

    @staticmethod
    def compare_exists_in_regions(measure_regions, predicted_regions):
        overlapping_matrix = np.zeros((len(measure_regions), len(predicted_regions)))
        for m in range(len(measure_regions)):
            for p in range(len(predicted_regions)):
                overlapping_matrix[m, p] = measure_regions[m].is_overlapping(predicted_regions[p])
        return overlapping_matrix


class CoverArea:
    @staticmethod
    def calculate(base_region, overlapping_regions):
        assert isinstance(base_region, Region)
        if base_region.h < 1 or base_region.w < 1:   # TODO: Prevent this situation
            pass
        area_boxes = np.zeros(shape=(base_region.h, base_region.w))
        for overlapping_region in overlapping_regions:
            assert isinstance(overlapping_region, Region)
            x_dependent = overlapping_region.x - base_region.x
            y_dependent = overlapping_region.y - base_region.y
            area_boxes[y_dependent:(y_dependent+overlapping_region.h),
                       x_dependent:(x_dependent+overlapping_region.w)] = 1
        return area_boxes.sum()


def get_regions_from_contours(contours):
    """Get boxes regions and regions from list of detected contours."""
    boxes = []
    measured_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_length_of_region and h >= min_length_of_region:
            measured_regions.append(Region(x, y, w, h))
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            box = np.int0(box)
            boxes.append(box)
    return boxes, measured_regions
