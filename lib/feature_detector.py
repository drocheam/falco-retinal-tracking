import numpy as np
import cv2 as cv


class FeatureDetector:

    def __init__(self, *args) -> None:
        """
        Initialize the feature detector.

        :param args: arguments for the ORB class
        """
        self.kp1 = None
        self.des1 = None

        # initialize feature detector
        self.fdet = cv.ORB_create(*args)

    def get_features(self, image: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Get features from an image.

        :param image: input image
        :return: keypoint array, descriptor list
        """
        return self.fdet.detectAndCompute(image, None)

    def init_reference(self, reference: np.ndarray) -> None:
        """
        Initialize the reference frame.

        :param reference: reference frame
        """
        self.kp1, self.des1 = self.get_features(reference)

    def compute(self, 
                img_curr:           np.ndarray, 
                min_matches:        int, 
                std_th:             float,
                max_matches_shown:  int)\
                        \
            -> tuple[bool, 
                     tuple[float, float], 
                     np.ndarray,
                     list,
                     np.ndarray, 
                     np.ndarray, 
                     list]:
        """
        Detect and calculate image shift.
        Return matching properties.

        :param img_curr: current image frame
        :param min_matches: minimum number of matches
        :param std_th: maximum standard deviation of matches
        :param max_matches_shown: maximum number of matches to display
        :return: if detection was successful, image shift tuple, 
            keypoint mask, list of good keypoints, keypoints reference, 
            keypoints current frame, bool list for matches to show
        """
        # init
        shift = None
        features_found = False
        mask = []
        matches_shown = []

        # keypoints in current image
        kp2, des2 = self.get_features(img_curr)

        # match descriptors
        # see https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        if des2 is not None and self.des1 is not None:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.des1, des2)
            good = matches

        # apply additional criteria
        if len(good) > min_matches:

            src_pts = np.array([self.kp1[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 1, 2)
            dst_pts = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 1, 2)

            # we don't need the full homography from cv.findHomography (8 degrees of freedom),
            # instead use estimateAffinePartial2D with only 4 degrees of freedom
            M, maskp = cv.estimateAffinePartial2D(src_pts, dst_pts, None, cv.RANSAC, 3, confidence=0.95)

            # check if minimum number of matches
            if M is not None and np.count_nonzero(maskp) > min_matches:

                mask_bool = maskp.ravel().astype(bool)
                src_pts2 = src_pts[mask_bool, 0]
                dst_pts2 = dst_pts[mask_bool, 0]

                # make sure standard deviation between point shifts is small, otherwise something is wrong
                if np.sqrt(np.sum(np.std(dst_pts2 - src_pts2, axis=0)**2)) < std_th:

                    shift = np.mean(dst_pts2 - src_pts2, axis=0)
                    features_found = True
                    mask = maskp
        
                    # limit shown matches to 'max_matches_shown'
                    mask2 = mask.copy()
                    subm = mask2[mask2.astype(bool)]
                    subm[max_matches_shown:] = 0
                    mask2[mask2.astype(bool)] = subm[:]
                    matches_shown = mask2.ravel().tolist()

        # nothing found
        if not features_found:
            good = []

        return features_found, shift, mask, good, self.kp1, kp2, matches_shown

