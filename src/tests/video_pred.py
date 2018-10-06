from time import sleep
import cv2

from tests.correct_regions import load_vatic_regions
from video_chooser import get_youtube_video_with_selected_quality


def test_of_vatic_parser(url, vatic_output_path):
    line_thickness = 2
    color_green = (0, 255, 0)
    video_capture = get_youtube_video_with_selected_quality(url)
    right_measures = load_vatic_regions(vatic_output_path)
    t = 0
    while cv2.waitKey(1) & 0xFF != ord('q'):
        if not video_capture.isOpened():
            print('Camera is ot open.')
            sleep(5)
        sleep(0.02)
        ret, frame = video_capture.read()

        if t < len(right_measures):
            for measure in right_measures[t]:
                frame = cv2.rectangle(frame, (measure.x, measure.y), (measure.x2, measure.y2),
                                      color_green, line_thickness)
        cv2.imshow('Video-mask', frame)
        t += 1
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main___":
    test_of_vatic_parser("https://www.youtube.com/watch?v=PNCJQkvALVc", "D:/Wiadomosci/vatic-output/output.xml")
