from time import sleep

import cv2
import numpy as np

from transformations import MorphologicalTransformer
from region import get_regions_from_contours
from tests.correct_regions import load_vatic_regions
from tracker.metrics import AllMetricWrapper
from tracker.multi_tracker import MultiObjectsTracker


# Parameters
max_object_count = 30
frames_count_to_stabilize = 20
max_frames_count_of_missing_track = 5
window_main_name = 'Video-Main'
path_correct_vatic_regions = "D:/Wiadomosci/vatic-output/output.xml"
n_mixtures = 4
learning_rate_of_subtractor = 0.000000001
font = cv2.FONT_HERSHEY_SIMPLEX
line_thickness = 2
font_scale = 1
yt_url_video = "https://www.youtube.com/watch?v=PNCJQkvALVc"  # M6 Motorway Traffic

show_correct_boxes = False
break_after_test_end = True
draw_all_founded_contours = False


def main(video_capture, show_intermediate_states=False):
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 7)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Chosen size:', width, 'x', height)
    cv2.namedWindow(window_main_name, cv2.WINDOW_NORMAL)

    morph_transformer = MorphologicalTransformer(window_main_name)
    morph_transformer.create_trackbars()
    multi_tracker = MultiObjectsTracker(lost_track_patience=max_frames_count_of_missing_track)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=16)
    background_subtractor.setNMixtures(n_mixtures)
    background_subtractor.setDetectShadows(True)
    # Quick start for background subtractor
    ret, frame = video_capture.read()
    background_subtractor.apply(frame, 0.01)

    # Tests
    if show_correct_boxes:
        right_regions = load_vatic_regions(path_correct_vatic_regions)
    all_metrics = AllMetricWrapper()

    frames_viewed = 0
    while cv2.waitKey(1) & 0xFF != ord('q'):

        if not video_capture.isOpened():
            print('Camera is ot open.')
            sleep(5)
            continue
        sleep(0.01)  # It's  slowdown video processing for more smooth viewing
        ret, frame = video_capture.read()
        frames_viewed += 1

        frame_blurred = morph_transformer.apply_blur(frame)
        mask_org = background_subtractor.apply(frame_blurred, learning_rate_of_subtractor)
        mask_transformed = apply_morphological_operations(morph_transformer, mask_org, show_intermediate_states)

        # Remove shadows
        mask_with_shadows = np.copy(mask_transformed)
        _, mask_transformed = cv2.threshold(mask_transformed, 128, 255, cv2.THRESH_BINARY)
        output_after_mask = cv2.bitwise_and(frame, frame, mask=mask_transformed)

        # Find, track and draw regions
        _, contours, hierarchy = cv2.findContours(mask_transformed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > max_object_count:
            count_str = str(len(contours))
            print("Warning: It's to many objects - probably noise. Check morphological operations.Count = " + count_str)
        else:
            color_green = (0, 255, 0)
            color_red = (0, 0, 255)
            color_white = (255, 255, 255)
            frame_with_regions = frame
            boxes, measured_regions = get_regions_from_contours(contours)

            if draw_all_founded_contours:
                all_boxes = -1
                frame_with_all_contours = cv2.drawContours(np.copy(frame), boxes, all_boxes, color_red, line_thickness)
                cv2.imshow("Video - only contours", frame_with_all_contours)

            predictions = []

            # Show correct regions - testing
            if show_correct_boxes and frames_viewed < len(right_regions):
                for measure in right_regions[frames_viewed]:
                    frame = cv2.rectangle(frame, (measure.x, measure.y), (measure.x2, measure.y2), color_green, 1)

            # Update tracker and show predicted regions
            if frames_viewed > frames_count_to_stabilize:
                print("Kalman tracker:" + str(len(multi_tracker.trackers)) +
                      ", regions=" + str(len(boxes)) + ", frame id =" + str(frames_viewed))
                print(multi_tracker.statistics())
                multi_tracker.update(measured_regions)
                predictions = multi_tracker.predictions

                # Draw statistics and predicted regions:
                frame = draw_predicted_regions(frame, multi_tracker)
                frame_with_regions = cv2.putText(
                    frame_with_regions, 'Objects: ' + str(len(boxes)), (22, 50),
                    font, font_scale, color_white, line_thickness, cv2.LINE_AA
                )
                frame_with_regions = cv2.putText(
                    frame_with_regions, 'Trackers: ' + str(len(multi_tracker.trackers)), (width - 240, 50),
                    font, font_scale, color_white, line_thickness, cv2.LINE_AA
                )
                cv2.imshow('Video-Tracked_regions', frame_with_regions)

            # Collect metrics
            if show_correct_boxes and frames_viewed < len(right_regions):
                all_metrics.update_metric_for_frame(right_regions[frames_viewed], predictions)
                print(all_metrics.calculate_result())
            elif break_after_test_end:
                break
        # Show additional windows:
        if show_intermediate_states:
            cv2.imshow('Video-mask', mask_transformed)
            cv2.imshow('Video-mask_and_shadows', mask_with_shadows)
            background = background_subtractor.getBackgroundImage()
            cv2.imshow('Video-background', background)
        cv2.imshow(window_main_name, output_after_mask)
    video_capture.release()
    cv2.destroyAllWindows()


def draw_predicted_regions(frame, multi_tracker):
    for tracker in multi_tracker.trackers:
        prediction = tracker.act_prediction
        frame = cv2.rectangle(frame, (prediction.x, prediction.y), (prediction.x2, prediction.y2),
                              tracker.color, line_thickness)
        frame = cv2.putText(frame, str(tracker.tracker_id), (prediction.x, prediction.y), font, 1,
                            tracker.color, line_thickness, cv2.LINE_AA)
    return frame


def apply_morphological_operations(morph_transformer, mask_org, show_intermediate_states):
    if show_intermediate_states:
        cv2.imshow('Video-mask-0.BeforeAll', mask_org)
    mask = morph_transformer.apply_erode(mask_org)
    if show_intermediate_states:
        cv2.imshow('Video-mask-1.After Erode', mask)
    mask = morph_transformer.apply_dilate(mask)
    if show_intermediate_states:
        cv2.imshow('Video-mask-2.After Dilate', mask)
    mask = morph_transformer.apply_morph_open(mask)
    mask = morph_transformer.apply_morph_close(mask)
    if show_intermediate_states:
        cv2.imshow('Video-mask-3.After Open or Close', mask)
    return mask


if __name__ == "__main__":
    # vid_capture = get_youtube_video_with_choice_quality(yt_url_video)
    vid_capture = cv2.VideoCapture("/home/deep-learning/Downloads/A3 A-road Traffic UK 480x360.mp4")
    main(vid_capture, show_intermediate_states=False)
