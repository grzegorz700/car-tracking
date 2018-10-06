import random
from queue import Queue

import cv2
import numpy as np

from region import Region


class KalmanTracker:
    """Kalman tracker designed for tracking a rectangle object witch changes
    its size and position. It uses a kalman filter. Default use consists of
    prediction the next state and update state based on actual measurement.
     """
    next_id = 1
    matrix_size = 8
    min_matrix_size = 4
    measure_params = 4
    max_history_predictions = 5

    def __init__(self, start_measure, meaningful_of_noise=0.5):
        self.tracker_id = KalmanTracker.next_id
        KalmanTracker.next_id += 1
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Set random RGB color
        self.start_measure = start_measure
        self.act_prediction = start_measure
        self.frames_without_update = 0
        self.life_time = 0
        self.prediction_history = Queue(KalmanTracker.max_history_predictions)

        self.kalman_filter = cv2.KalmanFilter(
            dynamParams=KalmanTracker.matrix_size,       # Dimension of the state
            measureParams=KalmanTracker.measure_params,  # Dimension of measurement
            controlParams=0)                             # Dimension of control vector

        # Init state
        pre_state = self.kalman_filter.statePre
        pre_state = KalmanTracker.__get_updated_measure_array(pre_state, self.start_measure)
        self.kalman_filter.statePre = pre_state
        self.kalman_filter.statePost = np.copy(pre_state)  # Conversion

        self.kalman_filter.measurementMatrix = cv2.setIdentity(self.kalman_filter.measurementMatrix)

        # Set transitionMatrix
        self.__measures_addiction = np.diag(
            np.ones(KalmanTracker.matrix_size-KalmanTracker.min_matrix_size, dtype=np.float32),
            k=KalmanTracker.min_matrix_size)
        matrix_A = np.diag(np.ones(KalmanTracker.matrix_size, dtype=np.float32)) + self.__measures_addiction
        self.kalman_filter.transitionMatrix = matrix_A

        self.kalman_filter.processNoiseCov *= meaningful_of_noise
        self.kalman_filter.measurementNoiseCov = cv2.setIdentity(
            self.kalman_filter.measurementNoiseCov, KalmanTracker.__cv2_scalar_all(0.1))
        self.kalman_filter.errorCovPost = cv2.setIdentity(
            self.kalman_filter.errorCovPost, KalmanTracker.__cv2_scalar_all(0.1))

    @staticmethod
    def __cv2_scalar_all(value):
        return np.full(shape=4, fill_value=value)

    @staticmethod
    def __get_updated_measure_array(array, measure):
        array[0, 0] = measure.x
        array[1, 0] = measure.y
        array[2, 0] = measure.w
        array[3, 0] = measure.h
        return array

    @staticmethod
    def __decapsulate_measure(matrix):
        """Subtract 4 parameters (x,y,w,h) from the 2D list to 1D."""
        return matrix[:KalmanTracker.measure_params, 0]

    def correct(self, measure):
        """Correct the model based on real measures."""
        measures_array = np.array([measure.x, measure.y, measure.w, measure.h], dtype=np.float32)
        estimated = self.kalman_filter.correct(measures_array)
        estimated = KalmanTracker.__decapsulate_measure(estimated)
        self.act_prediction = Region.from_matrix(estimated)
        return self.act_prediction

    def predict(self):
        """Predict next position and size of tracked object."""
        prediction_raw = self.kalman_filter.predict()
        self.kalman_filter.statePost = np.copy(self.kalman_filter.statePre)
        self.kalman_filter.errorCovPost = np.copy(self.kalman_filter.errorCovPre)
        prediction_decapsulated = KalmanTracker.__decapsulate_measure(prediction_raw)
        prediction_wrap = Region.from_matrix(prediction_decapsulated)
        self.act_prediction = prediction_wrap

        # Collect last history:
        if self.prediction_history.full():
            self.prediction_history.get()
        self.prediction_history.put(self.act_prediction)

        return self.act_prediction

    def mark_as_updated_in_this_frame(self):
        self.life_time += 1
        self.frames_without_update = 0

    def mark_as_not_updated_in_this_frame(self):
        self.life_time += 1
        self.frames_without_update += 1
