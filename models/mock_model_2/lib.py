
from random import random


def get_result():
    return {
        "fatigue": round(random() / 5 + 0.4, ndigits=5),
    }
