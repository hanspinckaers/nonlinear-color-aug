import numbers
import random

import numpy as np
import PIL
import torch
from scipy import interpolate


class NonLinearColorJitter(object):
    """Randomly map the color channels in a non-linear fashion.

    Motivation (copied from https://github.com/deepmind/multidim-image-augmentation):
        Color augmentation helps to teach the network desired robustness and helps to reduce
        overfitting. Standard color augmentations (contrast, brightness) are often implemented as
        linear transforms, and so will most likely be directly compensated by the input
        normalization of a network. So we want to have non-linear augmentations (like
        gamma-correction and the S-curves in Photoshop). Trying to combine these two and find a
        reasonable parameterization ended in a nightmare, so here is a more straight-forward
        alternative.

        Instead of finding a parameterization, we just define the contraints to the mapping function
        -- which is much easier and intuitive (for the examples we assume float gray values between
        0 and 1)

        - the new "black point" should be within a certain range (e.g., -0.1 to 0.1)
        - the new "white point" should be within a certain range (e.g., 0.9 to 1.1)
        - the function should be reasonable smooth
        - the slope of the function should be bounded (e.g., between 0.5 and 2.0)

        The algorithm first samples control points (here 5) and then computes the smooth function
        via cubic bspline interpolation

        - sample a random value from the "black range" for the control point at 0, 
          the new "black point"
        - sample a random value from the "white range" for the control point at 1, 
          the new "white point"
        - recursively insert a new control point between the existing ones. Sample its value such
          that the slope contraints to both neightbours are fulfilled
        - map color channels using the cubic bspline interpolation curves

    Args:
        white_point (float or tuple of float (min, max)): How much to jitter the maximum white value
            a new max white point is chosen uniformly from [max(0, 1 - white_point), 1 + white_point]
            or the given [min, max]. Should be non negative numbers.
        black_point (float or tuple of float (min, max)): How much to jitter the minimum black value
            a new max white point is chosen uniformly from [max(0, 1 - black_point), 1 + black_point]
            or the given [min, max]. Should be non negative numbers.
        slope (float or tuple of float (min, max)): How much the mapping function can deviate from
            the original curve. slope_range is chosen uniformly from [max(0, 1 - slope), 1 + slope]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, white_point=0.1, black_point=0.1, slope=0.5):

        self.white_point = self._check_input(white_point, 'white_point')
        self.black_point = self._check_input(black_point, 'black_point', center=0, bound=(-1, 1), clip_first_on_zero=False)
        self.slope = self._check_input(slope, 'slope')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def control_point_between_neighbours(left_neighbour, right_neighbour, slope):
        """Insert a new control point between left_neighbour and right_neighbour"""

        middle_x = (right_neighbour[0] - left_neighbour[0]) / 2
        
        min_slope = slope[0]
        max_slope = slope[1]

        max_y = min(left_neighbour[1]  + max_slope * middle_x, 
                    right_neighbour[1] - min_slope * middle_x)
        
        min_y = max(left_neighbour[1]  + min_slope * middle_x, 
                    right_neighbour[1] - max_slope * middle_x)
        
        y = random.uniform(min_y, max_y)
        x = left_neighbour[0] + middle_x
        
        return torch.tensor([x, y])

    @classmethod
    def create_lookup_table(cls, black_point, white_point, slope):
        """Create mapping vector for values 0-255

        Arguments are same as that of __init__.

        Returns: 
            Vector with 256 values, index is current image pixel, value is new pixel.
        """
        black_point = torch.tensor([0, random.uniform(black_point[0], black_point[1])])
        white_point = torch.tensor([1, random.uniform(white_point[0], white_point[1])])

        middle = cls.control_point_between_neighbours(black_point, white_point, slope)
        quarter = cls.control_point_between_neighbours(black_point, middle, slope)
        three_quarter = cls.control_point_between_neighbours(middle, white_point, slope)

        vector = torch.stack([black_point, quarter, middle, three_quarter, white_point]) 

        x = vector[:,0]
        y = vector[:,1]

        f = interpolate.interp1d(x, y, kind='quadratic')

        xnew = torch.arange(0, 1, 1/256)
        ynew = f(xnew)
        
        return ynew

    @classmethod
    def get_params(cls, white_point=0.1, black_point=0.1, slope=0.5):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        r_new = cls.create_lookup_table(black_point, white_point, slope)
        g_new = cls.create_lookup_table(black_point, white_point, slope)
        b_new = cls.create_lookup_table(black_point, white_point, slope)

        return [r_new, g_new, b_new]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        curves = self.get_params(self.white_point, self.black_point, self.slope)
        np_img = np.array(img)

        for i in range(3):
            new_curve = curves[i]
            new_values = new_curve[np_img[..., i]]
            new_values = new_values.clip(0, 1)
            np_img[..., i] = new_values * 255

        return PIL.Image.fromarray(np_img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'white_point={0}'.format(self.white_point)
        format_string += ', black_point={0}'.format(self.black_point)
        format_string += ', slope={0}'.format(self.slope)
        return format_string
