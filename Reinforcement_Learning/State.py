########################################################################
# This file is based on the TensorFlow Tutorials available at:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# Published under the MIT License. See the file LICENSE for details.
# Copyright 2017 by Magnus Erik Hvass Pedersen
########################################################################

import numpy as np
import scipy.ndimage

########################################################################
# Functions and classes for processing images from the game-environment
# and converting them into a state.

class GlobalState:
    """
    Used for processing raw image-frames from the game-environment.

    The image-frames are converted to gray-scale, resized, and then
    the background is removed using filtering of the image-frames
    so as to detect motions.

    This is needed because a single image-frame of the game environment
    is insufficient to determine the direction of moving objects.

    The original DeepMind implementation used the last 4 image-frames
    of the game-environment to allow the Neural Network to learn how
    to detect motion. This implementation could make it a little easier
    for the Neural Network to learn how to detect motion, but it has
    only been tested on Breakout and Space Invaders, and may not work
    for games with more complicated graphics such as Doom. This remains
    to be tested.
    """

    def __init__(self, image, global_state_img_size, decay=0.75):
        """

        :param image:
            First image from the game-environment,
            used for resetting the motion detector.

        :param decay:
            Parameter for how long the tail should be on the motion-trace.
            This is a float between 0.0 and 1.0 where higher values means
            the trace / tail is longer.
        """

        # Pre-process the image and save it for later use.
        # The input image may be 8-bit integers but internally
        # we need to use floating-point to avoid image-noise
        # caused by recurrent rounding-errors.
        self.global_state_img_size = global_state_img_size

        img = _pre_process_image(image, self.global_state_img_size)
        self.last_input = img.astype(np.float)

        # Set the last output to zero.
        self.last_output = np.zeros_like(img)

        self.decay = decay

    def process(self, image):
        """Process a raw image-frame from the game-environment."""

        # Pre-process the image so it is gray-scale and resized.
        img = _pre_process_image(image, self.global_state_img_size)

        # Subtract the previous input. This only leaves the
        # pixels that have changed in the two image-frames.
        img_dif = img - self.last_input

        # Copy the contents of the input-image to the last input.
        self.last_input[:] = img[:]

        # If the pixel-difference is greater than a threshold then
        # set the output pixel-value to the highest value (white),
        # otherwise set the output pixel-value to the lowest value (black).
        # So that we merely detect motion, and don't care about details.
        img_motion = np.where(np.abs(img_dif) > 20, 255.0, 0.0)

        # Add some of the previous output. This recurrent formula
        # is what gives the trace / tail.
        output = img_motion + self.decay * self.last_output

        # Ensure the pixel-values are within the allowed bounds.
        output = np.clip(output, 0.0, 255.0)

        self.last_output = output

        return output

    def get_state(self, img, remaining_path_img):
        """
        Get a state that can be used as input to the Neural Network.

        It is basically just the last input and the last output of the
        motion-tracer. This means it is the last image-frame of the
        game-environment, as well as the motion-trace. This shows
        the current location of all the objects in the game-environment
        as well as trajectories / traces of where they have been.
        """

        self.process(img)


        path = _pre_process_image(remaining_path_img, self.global_state_img_size, rgb_convert=False)

        path_idx = np.where(path > 0)

        path[path_idx] = 255

        # Stack the last input and output images.
        state = np.dstack([self.last_input, self.last_output, path])

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state

def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray


def _pre_process_image(image, global_state_img_size, rgb_convert=True):
    """Pre-process a raw image from the game-environment."""

    # Convert image to gray-scale.
    if rgb_convert:
        image = _rgb_to_grayscale(image)

    # Resize to the desired size using SciPy for convenience.
    img = scipy.misc.imresize(image, size=global_state_img_size, interp='bicubic')

    return img