#!/usr/bin/env python3
import frame
import cv2 as cv

class RelativeEstimator:
    def __init__(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = 6,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
        search_params = dict(checks = 50)
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def match_frames(self, frame1, frame2):
        matches =self.matcher.knnMatch(frame1.desc, frame2.desc, k = 2)
        good_matches = []
        matched_uvs = []
        for match in matches:
            if len(match) < 2:
                continue
            first, second = match
            if first.distance < (0.75 * second.distance):
                good_matches.append(first)
        good_matches.sort(key = lambda m : m.distance)
        
        for match in good_matches:
            matched_pt1 = frame1.kps[match.queryIdx].pt
            matched_pt2 = frame2.kps[match.trainIdx].pt
            uv1 = tuple(int(round(coord)) for coord in matched_pt1)
            uv2 = tuple(int(round(coord)) for coord in matched_pt2)
            matched_uvs.append((uv1, uv2))

        return good_matches, matched_uvs