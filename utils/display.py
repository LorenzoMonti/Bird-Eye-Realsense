
"""display.py
"""


import cv2


def open_window(window_name, width, height, title):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowTitle(window_name, title)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 40), font, 3.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 40), font, 3.0, (240, 240, 240), 1, line)
    return img


def set_display(window_name, full_scrn):
    """Set display window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)
