from cfg import media_folder, images, descriptor

import time, sys, json

import cv2, numpy as np

##################################################
# precalculation
##################################################

etalon = cv2.imread(f'{media_folder}/frame0.jpg', cv2.IMREAD_COLOR)
etalon_kp, etalon_des = descriptor.compute(etalon)

##################################################
# calculation
##################################################

result = {}
for frame_name, ranges in images.items():
    result[frame_name] = {}
    for range_name, range_arr in ranges.items():
        match_results, distances, times = [], [], []
        for start, end in range_arr:
            for i in range(start, end):
                try:
                    t1 = time.time()

                    kp, des = descriptor.compute(cv2.imread(f'{media_folder}/{frame_name}{i}.jpg', cv2.IMREAD_COLOR))
                    matches = descriptor.match(etalon_des, des)

                    t2 = time.time()

                    ##################################################
                    # metrics collection
                    ##################################################

                    # 1
                    times.append(t2 - t1)

                    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    dist = sum(m.distance for m in good_matches)
                    source_points = np.float32([etalon_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dest_points = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    matchesMask = mask.ravel().tolist()
                    matchesFinal = [a for a,b in zip(good_matches,matchesMask) if b]

                    # 2
                    match_results.append(len(matchesFinal)/len(good_matches))
                    # 3
                    distances.append(dist/len(good_matches))
                except Exception as e:
                    print(f'Bad image ({i}):\n{e}', file=sys.stderr)

        result[frame_name][range_name] = { 
            'distance' : sum(distances) / len(distances),
            'time' : sum(times) / len(times),
            'features_amount_ratio' :  sum(match_results) / len(match_results),
        }

# dump structured results to output
print(json.dumps(result, indent=4))
