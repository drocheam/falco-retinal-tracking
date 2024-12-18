import cv2 as cv
import numpy as np

# some plotting helper functions

def new_window(name: str, size: list[int, int]) -> None:
    """
    Create and open the openCV window

    :param name: name of window
    :param size: size of window as [width, height]
    """
    cv.namedWindow(name, cv.WINDOW_NORMAL) 
    cv.waitKey(50)
    cv.resizeWindow(name, *size)
    cv.waitKey(50)

def keyboard_interaction() -> bool:
    """
    Handles the keyboard interaction with the cv window.
    Space to pause the stream, space to resume.
    q to exit

    :return: if the program should exit outside this function
    """
    key = cv.waitKey(2) & 0xFF

    if key == ord(' '):
        
        key = 0
        while key not in [ord(' '), ord('q')]:
            key = cv.waitKey(10) & 0xFF

    return key == ord('q')


def wait_for_exit(window_name: str) -> None:
    """
    Blocks the program (and therefore leaves the plot window open)
    until either 'q' is pressed or the window with 'window_name' gets closed

    :param window_name: name of the cv window to monitor
    """
    # leave window open, exit with q
    key = 0
    while key != ord('q') and cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) > 0:
        key = cv.waitKey(10) & 0xFF


def draw_found_position(img_hist:   np.ndarray, 
                        shift:      tuple[float, float], 
                        color:      tuple = None)\
        -> None:
    """
    Draw the detected position on the history image

    :param img_hist: history image, same shape as img_ref
    :param shift: image shift x value, y value
    :param color: color of point to draw as rgb tuple
    """

    # OCT beam in coordinates of reference image
    xp = round(img_hist.shape[1]/2 - shift[0])
    yp = round(img_hist.shape[0]/2 - shift[1])

    # inside reference image
    if 0 < xp < img_hist.shape[1] and 0 < yp < img_hist.shape[0]:
        cv.circle(img_hist, (xp, yp), 2, color, -1, cv.LINE_AA)


def draw_oct_beam_reference(img_comb:   np.ndarray, 
                            img_ref:    np.ndarray,
                            shift:      tuple[float, float])\
        -> None:
    """
    Draw the OCT beam in the reference image.

    :param img_comb: combined image of img_ref and current frame
    :param img_ref: reference image
    :param shift: image shift x value, y value
    """
    # OCT beam in coordinates of reference image
    xp = round(img_ref.shape[1]/2 - shift[0])
    yp = round(img_ref.shape[0]/2 - shift[1])

    # inside reference image
    if 0 < xp < img_ref.shape[1] and 0 < yp < img_ref.shape[0]:
        cv.drawMarker(img_comb, (xp, yp),  (255, 255, 255), cv.MARKER_CROSS, 21, 8);
        cv.drawMarker(img_comb, (xp, yp),  (0, 0, 255), cv.MARKER_CROSS, 14, 2);


def draw_current_oct_beam(img_comb: np.ndarray) -> None:
    """
    Draw the OCT beam position in the current frame.

    :param img_comb: combined image of img_ref and img_curr
    """
    xp2 = round(img_comb.shape[1] * 3 / 4)
    yp2 = img_comb.shape[0] // 2
    cv.drawMarker(img_comb, (xp2, yp2),  (255, 255, 255), cv.MARKER_CROSS, 21, 8);
    cv.drawMarker(img_comb, (xp2, yp2),  (0, 0, 255), cv.MARKER_CROSS, 14, 2);

