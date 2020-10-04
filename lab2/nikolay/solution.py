import time, json, cv2

media_folder = 'media'

images = {
    # photos where object is in frame
    'frame' : {
        # ranges of samples
        'original' : ((1, 30), (60, 100), (150, 212)), 
        'scaled60percents' : ((30, 60), ),
        'scaled200percents' : ((100, 150), ),
    },
    # photos with object out of frame
    'noframe' : {
        # ranges
        'original' : ((1, 36), )
    },
}

# initial setup
etalon = cv2.imread(f'{media_folder}/frame0.jpg', cv2.IMREAD_COLOR)
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# key points
etalon_kp = star.detect(etalon, None)

# descriptor
etalon_kp, etalon_des = brief.compute(etalon, etalon_kp)

result = {}
for frame_name, ranges in images.items():
    for range_name, range_arr in ranges.items():
        match_results, distances, times = [], [], []
        matched_num, kp_num = 0, 0
        for start, end in range_arr:
            for i in range(start, end):
                t1 = time.time()

                # matching code
                img = cv2.imread(f'{media_folder}/{frame_name}{i}.jpg', cv2.IMREAD_COLOR)
                kp = star.detect(img, None)
                kp, des = brief.compute(img, kp)
                matches = matcher.match(etalon_des, des)
                
                t2 = time.time()

                # metrics collection
                times.append(t2 - t1)
                distances.extend(x.distance for x in matches)
                matched_num += len(matches)
                kp_num  += len(kp)

        result.setdefault(frame_name, {})
        result[frame_name][range_name] = { 
            'distance' : sum(distances) / len(distances),
            'time' : sum(times) / len(times),
            'features_amount' :  kp_num / matched_num,
        }

print(json.dumps(result, indent=4))
