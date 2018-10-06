import numpy as np
import math

from region import Region, CoverArea


class FragmentationMetric:
    def __init__(self):
        self.fragmentation_metrics = []

    def update_metric_for_frame(self, correct_regions, predicted_regions):
        overlapping_matrix = Region.compare_exists_in_regions(correct_regions, predicted_regions)
        regions_count = len(correct_regions)
        count_of_predicted_regions_by_region = np.sum(overlapping_matrix, axis=1)
        for r in range(regions_count):
            if count_of_predicted_regions_by_region[r] > 0:
                overlapping_count = count_of_predicted_regions_by_region[r]
                metric = 1.0/(1.0 + math.log(overlapping_count))
                self.fragmentation_metrics.append(metric)

    def calculate_result(self):
        if len(self.fragmentation_metrics) > 0:
            return sum(self.fragmentation_metrics) / len(self.fragmentation_metrics)
        else:
            return None


class AverageObjectAreaRecallMetric:
    def __init__(self):
        self.recall_aor_metrics = []

    def update_metric_for_frame(self, correct_regions, predicted_regions):
        overlapping_matrix = Region.compare_exists_in_regions(correct_regions, predicted_regions)
        predicted_regions_count = len(predicted_regions)
        correct_regions_count = len(correct_regions)

        for cr in range(correct_regions_count):
            correct_region = correct_regions[cr]
            assert isinstance(correct_region, Region)
            overlapping_regions = []

            for pr in range(predicted_regions_count):
                if overlapping_matrix[cr, pr]:
                    full_region_overlapping = predicted_regions[pr]
                    overlapping_region = correct_region.get_own_subregion(full_region_overlapping)
                    overlapping_regions.append(overlapping_region)

            overlapping_area = CoverArea.calculate(correct_region, overlapping_regions)
            metric = overlapping_area / correct_region.get_area()
            self.recall_aor_metrics.append(metric)

    def calculate_result(self):
        if len(self.recall_aor_metrics) > 0:
            return sum(self.recall_aor_metrics) / len(self.recall_aor_metrics)
        else:
            return None


class AverageDetectedBoxAreaPrecisionMetric:
    def __init__(self):
        self.precision_adba_metrics = []

    def update_metric_for_frame(self, correct_regions, predicted_regions):
        overlapping_matrix = Region.compare_exists_in_regions(correct_regions, predicted_regions)
        predicted_regions_count = len(predicted_regions)
        correct_regions_count = len(correct_regions)

        for pr in range(predicted_regions_count):
            predicted_region = predicted_regions[pr]
            assert isinstance(predicted_region, Region)
            overlapping_regions = []

            for cr in range(correct_regions_count):
                if overlapping_matrix[cr, pr]:
                    full_region_overlapping = correct_regions[cr]
                    overlapping_region = predicted_region.get_own_subregion(full_region_overlapping)
                    overlapping_regions.append(overlapping_region)

            overlapping_area = CoverArea.calculate(predicted_region, overlapping_regions)
            predicted_area = predicted_region.get_area()
            if predicted_area > 0:
                metric = overlapping_area / predicted_area
            else:
                metric = 0
            self.precision_adba_metrics.append(metric)

    def calculate_result(self):
        if len(self.precision_adba_metrics) > 0:
            return sum(self.precision_adba_metrics) / len(self.precision_adba_metrics)
        else:
            return None


class AllMetricWrapper:
    def __init__(self):
        self.fragmentation_metric = FragmentationMetric()
        self.recall_metric = AverageObjectAreaRecallMetric()
        self.precision_metric = AverageDetectedBoxAreaPrecisionMetric()
        self.__all_metrics = [self.fragmentation_metric, self.recall_metric, self.precision_metric]

    def update_metric_for_frame(self, correct_regions, predicted_regions):
        for metric in self.__all_metrics:
            metric.update_metric_for_frame(correct_regions, predicted_regions)

    def calculate_result(self):
        results = []
        for metric in self.__all_metrics:
            results.append(metric.calculate_result())
        return results
