import numpy as np

from tracker.kalman import KalmanTracker
from region import Region


class MultiObjectsTracker:
    """Track multiple rectangular objects using and managing Kalman tracker.
    Logic of tracker management based on the doctoral dissertation (in Polish language):
        Dalka, P. (2015). Metody algorytmicznej analizy obrazu wizyjnego do zastosowań w monitorowaniu ruchu drogowego
    """

    def __init__(self, lost_track_patience, min_prediction_area=10):
        self.min_prediction_area = min_prediction_area
        self.lost_track_patience = lost_track_patience
        self.trackers = []
        self.predictions = []

        # Collect statistics of 6 types of situation
        self.cs1 = CaseStatistic("Tracker  1  <=> 0  Regions", "T:1-0:R")
        self.cs2 = CaseStatistic("Trackers 0  <=> 1  Region",  "T:0-1:R")
        self.cs3 = CaseStatistic("Trackers 1  <=> 1  Region",  "T:1-1:R")
        self.cs4 = CaseStatistic("Trackers 2+ <=> 1  Region",  "T:2+-1:R")
        self.cs5 = CaseStatistic("Trackers 1  <=> 2+ Regions", "T:1-2+:R")
        self.cs6 = CaseStatistic("Trackers 2+ <=> 2+ Regions", "T:2+-2+:R")

    def update(self, measures_regions):
        if len(measures_regions) == 0:
            self.update_in_nothing_detected_region_case()
            return  # !!!

        # Create tracker for all new regions
        if len(self.trackers) == 0:
            for measure in measures_regions:
                self.trackers.append(KalmanTracker(measure))

        # Collect predictions
        predicted_regions = []
        trackers_to_remove = []
        for tracker in self.trackers:
            # Update predictions
            tracker.predict()
            # Delete when predicted region width or height change its value to negative or min area is too small
            if tracker.act_prediction.was_reversed or tracker.act_prediction.get_area() < self.min_prediction_area:
                trackers_to_remove.append(tracker)
            else:
                predicted_regions.append(tracker.act_prediction)
        for tracker_to_remove in trackers_to_remove:
            self.trackers.remove(tracker_to_remove)

        # Get overlapping area of regions.
        overlapping_matrix = Region.compare_exists_in_regions(measures_regions, predicted_regions)

        trackers_count = len(self.trackers)
        regions_count = len(measures_regions)
        trackers_count_by_region = np.sum(overlapping_matrix, axis=1)
        regions_count_by_tracker = np.sum(overlapping_matrix, axis=0)

        # Update predictions for 6 different cases.
        # 1. Trackers 1  <=> 0  Regions
        for i in range(trackers_count):
            if regions_count_by_tracker[i] == 0:
                self.trackers[i].mark_as_not_updated_in_this_frame()
                self.cs1.count += 1

        # 2. Trackers 0  <=> 1  Region
        new_trackers = []
        for i in range(regions_count):
            if trackers_count_by_region[i] == 0:
                tracker = KalmanTracker(measures_regions[i])
                tracker.correct(measures_regions[i])
                new_trackers.append(tracker)
                self.cs2.count += 1

        for t in range(trackers_count):
            if regions_count_by_tracker[t] == 1:
                for r in range(regions_count):
                    if overlapping_matrix[r, t]:
                        # 3. Trackers 1  <=> 1  Regions
                        if trackers_count_by_region[r] == 1:
                            tracker = self.trackers[t]
                            tracker.correct(measures_regions[r])
                            tracker.mark_as_updated_in_this_frame()
                            self.cs3.count += 1
                        # 4. Trackers 2+ <=> 1  Regions
                        elif trackers_count_by_region[r] > 1:
                            base_region = measures_regions[r]
                            tracker_region = predicted_regions[t]
                            sub_region = base_region.get_own_subregion(tracker_region)
                            self.trackers[t].correct(sub_region)
                            self.trackers[t].mark_as_updated_in_this_frame()
                            self.cs4.count += 1
            elif regions_count_by_tracker[t] > 1:
                # 5.&6. collect regions
                regions_for_tracker = []
                nobody_have_multi_trackers = True
                for r in range(regions_count):
                    if overlapping_matrix[r, t]:
                        if trackers_count_by_region[r] == 1:
                            regions_for_tracker.append(measures_regions[r])
                        elif trackers_count_by_region[r] > 1:
                            regions_for_tracker.append(measures_regions[r])
                            nobody_have_multi_trackers = False

                if len(regions_for_tracker) > 0:
                    merged_region = Region.merge_regions(regions_for_tracker)
                    # 5. Trackers 1  <=> 2+ Regions - only if no regions have other tracker!
                    if nobody_have_multi_trackers:
                        self.trackers[t].correct(merged_region)
                        self.trackers[t].mark_as_updated_in_this_frame()
                        self.cs5.count += 1
                    # 6. Trackers 2+ <=> 2+ Regions
                    else:
                        sub_region = merged_region.get_own_subregion(predicted_regions[t])
                        self.trackers[t].correct(sub_region)
                        self.trackers[t].mark_as_updated_in_this_frame()
                        self.cs6.count += 1

        # Delete outdated tracker:
        self.trackers = [tracker for tracker in self.trackers
                         if tracker.frames_without_update < self.lost_track_patience]

        self.trackers.extend(new_trackers)
        self.predictions = [tracker.act_prediction for tracker in self.trackers if tracker.frames_without_update == 0]

    def update_in_nothing_detected_region_case(self):
        trackers_to_remove = []
        for tracker in self.trackers:
            tracker.mark_as_not_updated_in_this_frame()
            # Remove missing tracker
            if tracker.frames_without_update > self.lost_track_patience:
                trackers_to_remove.append(tracker)
        for tracker in trackers_to_remove:
            self.trackers.remove(tracker)
        # Predict the move in hidden:
        for tracker in self.trackers:
            predict = tracker.predict()
        self.predictions = [prediction.act_prediction for prediction in self.trackers]

    def statistics(self):
        statistics = str([self.cs1, self.cs2, self.cs3, self.cs4, self.cs5, self.cs6])
        return str(len(self.trackers))+" Trackers, cases = " + statistics


class CaseStatistic:
    def __init__(self, name, short_name=None, count=0):
        self.short_name = short_name if short_name is not None else name
        self.count = count
        self.name = name

    def __repr__(self):
        return "(" + self.short_name + " -> " + str(self.count) + ")"
