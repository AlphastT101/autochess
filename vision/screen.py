import cv2
import numpy as np
import mss

def grab_screenshot() -> np.ndarray:
    """Full-screen screenshot as a BGR numpy array."""
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
