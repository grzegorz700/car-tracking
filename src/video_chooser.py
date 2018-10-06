import pafy
import cv2


def get_youtube_video_with_selected_quality(url):
    assert url is not None
    video_pafy = pafy.new(url)
    streams = video_pafy.allstreams
    name_of_qualities = [str(s) for s in streams]
    for ind in range(len(name_of_qualities)):
        print(str(ind) + '.', name_of_qualities[ind])
    chosen_quality_ind = input("Choose quality:")
    our_live_stream = streams[int(chosen_quality_ind)]
    video_capture = cv2.VideoCapture(our_live_stream.url)
    return video_capture


def get_video_from_file(path):
    video_capture = cv2.VideoCapture(path)
    return video_capture


def get_video_from_camera():
    video_capture = cv2.VideoCapture(0)
    return video_capture
