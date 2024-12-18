import cv2 as cv
import numpy as np


class VideoStream:

    def __init__(self, path: str, hsize: int = None):
        """
        Initialize the video stream.
        The output width is specified, the height is rescaled to match the initial aspect ratio.

        :param path: path to video / device
        :param hsize: horizontal size of the video
        """
        # video capture object
        self.cap = cv.VideoCapture(path, cv.CAP_ANY)

        # Check if the video was opened successfully
        if not self.cap.isOpened():
            raise IOError(f"Unable to open file '{path}'")

        # store frame rate and size parameters
        self.fps_org = self.cap.get(cv.CAP_PROP_FPS)
        size0 = list((int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),\
                      int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
        self.hsize = hsize
        self.size = size0 if hsize is None else [hsize, round(hsize*size0[1]/size0[0])]

    def get_frame(self) -> np.ndarray | None:
        """
        Get the next frame.

        :return: greyscale frame (np.float32). 'None' if reading failed
        """
        # read frame
        frame_read, frame = self.cap.read()

        # reading failed
        if not frame_read:
            return None

        # use only one channel (=grayscale)
        frame = frame[:, :, 0]

        # rescale the frame with INTER_AREA option for averaging and less aliasing
        if self.hsize is not None:
            frame = cv.resize(frame.astype(np.float32), self.size, interpolation=cv.INTER_AREA)

        # convert to float32
        return frame.astype(np.float32)

    def set_position(self, seconds: float) -> None:
        """
        Set the current video stream position.

        :param seconds: timestamp in seconds
        """
        self.cap.set(cv.CAP_PROP_POS_MSEC, seconds*1000)

    def close(self) -> None:
        """
        Close the video stream.
        """
        self.cap.release()

