# Vehicles tracking using a static traffic camera.

This is a student project for Computer Vision.

![Tracking example](example.png)

## Main components

1. Video choosing between local video and youtube.
1. Background subrataction with MOG (Gaussian Mixture-based Background/Foreground Segmentation Algorithm)
1. Morphological transformations with shadows removing.
1. Contours detection.
1. Trackers management (by adding, updating, removing them). Trackers base on Kalman Filters.


## Technology
- Python 3
- OpenCV 3
- VATIC - Video Annotation Tool (for evaluation)

## Plans:
- [ ] Combining all windows to one window (in progress)
- [ ] Improving methodology of  trackers management (from the [2] to my sth better)


## References
- [1] Bishop, G., & Welch, G. (2001). An introduction to the kalman filter. Proc of SIGGRAPH, Course, 8(27599-23175), 41.
- [2] Dalka, P. (2015). Metody algorytmicznej analizy obrazu wizyjnego do zastosowań w monitorowaniu ruchu drogowego (Rozprawa doktorska).
- [3] Dalka, P. .Detekcja i śledzenie ruchomych obiektów w obrazie. https://sound.eti.pg.gda.pl/student/pdio/pdio05_detekcja_sledzenie_obiektow.pdf
- [4] Johnsen, S., & Tews, A. (2009, May). Real-time object tracking and classification using a static camera. In Proceedings of IEEE International Conference on Robotics and Automation, workshop on People Detection and Tracking.
- [5] KaewTraKulPong, P., & Bowden, R. (2002). An improved adaptive background mixture model for real-time tracking with shadow detection. Video-based surveillance systems, 1, 135-144.
- [6] Leibe, B., Schindler, K., Cornelis, N., & Van Gool, L. (2008). Coupled object detection and tracking from static cameras and moving vehicles. IEEE transactions on pattern analysis and machine intelligence, 30(10), 1683-1698.
- [7] Roshani K. Dharme, Dipali Y. Shahare(2016) Moving Object Detection with Static and Dynamic Camera for Automated Video. Analysis. Journal of information, knowledge and research in computer engineering.
- [8] Sivaraman, S., & Trivedi, M. M. (2013). Looking at vehicles on the road: A survey of vision-based vehicle detection, tracking, and behavior analysis. IEEE Transactions on Intelligent Transportation Systems, 14(4), 1773-1795.
- [9] Zivkovic, Z. (2004, August). Improved adaptive Gaussian mixture model for background subtraction. In Pattern Recognition, 2004. ICPR 2004. Proceedings of the 17th International Conference on (Vol. 2, pp. 28-31). IEEE.

The most comprehensive source is doctoral dissertation [2], but it's only in Polish.