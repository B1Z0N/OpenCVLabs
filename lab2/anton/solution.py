import time

import cv2

media_folder = 'media'

images = {
    # photos where object is in frame
    'frame' : {
        # ranges of samples
        'original' : ((1, 10),(30,50),(70,110)), 
        'scaled40percents' : ((10, 30), ),
        'scaled200percents' : ((50, 70), ),
    },
    # out of frame
    'noframe' : {
        # ranges
        'original' : ((111, 120), ),
    },
}

etalon = cv2.imread(f'{media_folder}/frame0.JPG', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(etalon, None)


result = {}
for frame_name, ranges in images.items():
    for range_name, range_arr in ranges.items():
        distances, times = [], []
        matched_num, img_kp_num = 0, 0
        for start, end in range_arr:
            for i in range(start, end):
                start_t = time.time()
                try:
                    start_t = time.time()

                    img = cv2.imread(f'{media_folder}/{frame_name}{i}.JPG', cv2.IMREAD_GRAYSCALE)
                    kp2, des2 = sift.detectAndCompute(img,None)
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1,des2,k=2)
                    
                    end_t = time.time()
                    matched_num += len(matches)
                    img_kp_num  += len(kp2)
                    distances.extend(x.distance for x, y in matches if x.distance < 0.75*y.distance)
                    times.append(end_t - start_t)
                except:
                    # print('Bad image ' + str(i))
                    pass
                    

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