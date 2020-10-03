import time

import cv2

media_folder = 'media'

images = {
    # photos where object is in frame
    'frame' : {
        # ranges of samples
        'original' : ((1, 30), (60, 100), (150, 212)), 
        'scaled60percents' : ((30, 60), ),
        'scaled200percents' : ((100, 150), ),
    },
    # out of frame
    'noframe' : {
        # ranges
        'original' : ((1, 36), )
    },
}

etalon = cv2.imread(f'{media_folder}/frame0.jpg', cv2.IMREAD_COLOR)
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Detect the CenSurE key points
etalon_kp = star.detect(etalon, None)

# compute the descriptors
etalon_kp, etalon_des = brief.compute(etalon, etalon_kp)

result = {}
for frame_name, ranges in images.items():
    for range_name, range_arr in ranges.items():
        match_results, distances, times = [], [], []
        matched_num, img_kp_num = 0, 0
        for start, end in range_arr:
            for i in range(start, end):
                start_t = time.time()

                img = cv2.imread(f'{media_folder}/{frame_name}{i}.jpg', cv2.IMREAD_COLOR)
                img_kp = star.detect(img, None)
                img_kp, img_des = brief.compute(img, img_kp)
                matches = matcher.match(etalon_des, img_des)
                
                end_t = time.time()
                times.append(end_t - start_t)
                distances.extend(x.distance for x in matches)
                matched_num += len(matches)
                img_kp_num  += len(img_kp)

        result.setdefault(frame_name, {})
        result[frame_name][range_name] = { 
            'distance' : sum(distances) / len(distances),
            'time' : sum(times) / len(times),
            'features_amount' :  img_kp_num / matched_num,
        }


import json
print(json.dumps(result, indent=4))

# Paint the key points over the original image

# matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:20], None)

# cv2.imshow('Match', matches)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
