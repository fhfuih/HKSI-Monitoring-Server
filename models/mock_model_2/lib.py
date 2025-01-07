from random import random


def get_result(someParam):
    return {
        "fatigue": someParam,
        "pimpleCount": 2,
        "darkCircleLeft": True,
        "darkCircleRight": False,
        "darkCircles": {
            "left": [(0, 0), (0.5, 0.5), (0, 0.5)],
            "right": None,
        },
        "pimples": {
            "count": 2,
            "coordinates": [(0.5, 0.5, 0.2, 0.2), (0.3, 0.3, 0.1, 0.1)],
        },
    }
