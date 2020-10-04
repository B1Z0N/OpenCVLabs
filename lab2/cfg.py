import os
from abc import ABC, abstractmethod, abstractstaticmethod

import cv2

##################################################
# descriptors abstraction
##################################################

class Descriptor(ABC):
    def __init__(self):
        self.bf = cv2.BFMatcher(self.norm())

    @abstractmethod
    def compute(self, image):
        pass 

    @abstractstaticmethod
    def norm():
        pass

    def match(self, des1, des2):
        return self.bf.knnMatch(des1,des2, k=2)


class SIFT(Descriptor):
    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create()

    def compute(self, image):
        return self.sift.detectAndCompute(image, None)

    @staticmethod
    def norm():
        return cv2.NORM_L2


class BRIEF(Descriptor):
    def __init__(self):
        super().__init__()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.star = cv2.xfeatures2d.StarDetector_create()

    def compute(self, image):
        kp = self.star.detect(image, None)
        return self.brief.compute(image, kp)

    @staticmethod
    def norm():
        return cv2.NORM_HAMMING

##################################################
# tweakable part
##################################################

OWNERS_DISTINCTIONS = {
    'anton' : {
        'images' : {
            # photos where object is in frame
            'frame' : {
                # ranges of samples
                'original' : ((1, 10),(30,50),(70,110), ), 
                'scaled40percents' : ((10, 30), ),
                'scaled200percents' : ((50, 70), ),
            },
            # out of frame
            'noframe' : {
                # ranges
                'original' : ((111, 120), ),
            },
        },
        'descriptor' : SIFT
    },
    'nikolay' : {
        'images' : {
            # photos where object is in frame
            'frame' : {
                # ranges of samples
                'original' : ((1, 30), (60, 100), (150, 212), ), 
                'scaled60percents' : ((30, 60), ),
                'scaled200percents' : ((100, 150), ),
            },
            # photos with object out of frame
            'noframe' : {
                # ranges
                'original' : ((1, 36), )
            },
        },
        'descriptor' : BRIEF
    },
}

##################################################
# env vars logic
##################################################

ALLOWED_OWNERS = set(OWNERS_DISTINCTIONS.keys())

MEDIA_OWNER_VARNAME, CODE_OWNER_VARNAME = 'MEDIA_OWNER', 'CODE_OWNER'
whose_media, whose_code = os.getenv(MEDIA_OWNER_VARNAME), os.getenv(CODE_OWNER_VARNAME)

for var in (whose_media, whose_code):
    if var is None or var not in ALLOWED_OWNERS:
        raise EnvironmentError(
            f"""
    You should setup each of  MEDIA_OWNER and CODE_OWNER environment variables
    to either one of this {list(ALLOWED_OWNERS)}
""")

##################################################
# resulting variables making distinction 
# in code and data
##################################################

media_folder = f'media/{whose_media}' 
images = OWNERS_DISTINCTIONS[whose_media]['images']
descriptor = OWNERS_DISTINCTIONS[whose_code]['descriptor']()
