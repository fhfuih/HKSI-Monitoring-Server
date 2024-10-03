from random import random


def get_result():
    return {
        "fatigue": round(random() / 10 + 0.2, ndigits=3),
        "darkCircles": {
            "count": 0,
        },
        "pimples": {
            "count": 2,
        },
    }
