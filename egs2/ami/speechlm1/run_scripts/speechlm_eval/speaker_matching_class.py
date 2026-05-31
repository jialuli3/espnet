import numpy as np
from scipy.optimize import linear_sum_assignment
import pdb

class SpeakerSegmentMatcher:
    def __init__(self, windows, threshold=0.2, iou_threshold=0.01, merge_tolerance=0.2, skip=10):
        self.windows = windows
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.merge_tolerance = merge_tolerance
        self.skip = skip
        self.out_windows = {}

    @staticmethod
    def compute_overlap(segment1, segment2):
        start1, end1 = segment1
        start2, end2 = segment2
        return max(0.0, min(end1, end2) - max(start1, start2))

    def compute_overlap_matrix(self, segments_a, segments_b):
        num_a, num_b = len(segments_a), len(segments_b)
        matrix = np.zeros((num_a, num_b))

        for i, segs_a in enumerate(segments_a):
            for j, segs_b in enumerate(segments_b):
                matrix[i, j] = -sum(
                    self.compute_overlap(sa, sb)
                    for sa in segs_a for sb in segs_b
                )
        return matrix

    def match_speakers(self, window_a, window_b):
        speakers_a = list(window_a.keys())
        speakers_b = list(window_b.keys())
        segs_a = [window_a[s] for s in speakers_a]
        segs_b = [window_b[s] for s in speakers_b]

        cost_matrix = self.compute_overlap_matrix(segs_a, segs_b)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        #return {speakers_b[j]: speakers_a[i] for i, j in zip(row_ind, col_ind)}
        return {speakers_a[i]: speakers_b[j] for i, j in zip(row_ind, col_ind)}

    @staticmethod
    def compute_iou(s1, e1, s2, e2):
        union = max(e1, e2) - min(s1, s2)
        intersect = max(min(e1, e2) - max(s1, s2), 0)
        return intersect / union

    @staticmethod    
    def compute_overlap_ratio(s1, e1, s2, e2):
        overlap = max(0, min(e1, e2) - max(s1, s2))
        coverage = overlap / (e1 - s1)
        return coverage

    @staticmethod
    def get_overlap_interval(s1, e1, s2, e2):
        if s1 <= e2 and s2 <= e1:
            return max(s1, s2), min(e1, e2)
        return None

    def get_non_overlap_regions(self, intervals):
        events = []

        for start, end in intervals:
            events.append((start, 'start'))
            events.append((end, 'end'))

        events.sort()

        coverage = 0
        last_point = None
        covered = []
        unique_regions = []

        for point, kind in events:
            if last_point is not None and point != last_point:
                if coverage == 1:
                    # Only one interval covered this region
                    unique_regions.append((last_point, point))
            if kind == 'start':
                coverage += 1
            else:
                coverage -= 1
            last_point = point
        return unique_regions

    def stitch_segments(self):
        all_keys = self.windows.keys()
        segments_keys = {}
        iou_values = {}
        segments_intervals, segments_non_ovl_intervals = {}, {}
        for i,curr_key in enumerate(all_keys):
            curr_id = curr_key.split("-")[0]
            curr_interval = (int(curr_key.split("-")[1]), int(curr_key.split("-")[2]))
            segments_keys.setdefault(curr_id, []).append(curr_key)
            self.out_windows.setdefault(curr_id, {})
            segments_intervals.setdefault(curr_id, []).append(curr_interval)
        
        for curr_id in segments_intervals:
            segments_non_ovl_intervals[curr_id]=self.get_non_overlap_regions(segments_intervals[curr_id])
            # print(curr_id, segments_non_ovl_intervals[curr_id])
            
        for curr_id in segments_keys:
            curr_all_keys = sorted(segments_keys[curr_id], key=lambda x: int(x.split('-')[1]))
            for i in range(len(curr_all_keys) - 1):
                key_1, key_2 = curr_all_keys[i], curr_all_keys[i + 1]
                start = int(key_1.split("-")[1])
                matched = self.match_speakers(self.windows[key_1], self.windows[key_2])
                
                for spk in self.windows[key_1]:
                    for s1, e1 in self.windows[key_1][spk]:
                        for non_ovl_start, non_ovl_end in segments_non_ovl_intervals[curr_id]:
                            if overlap := self.get_overlap_interval(non_ovl_start, non_ovl_end, s1, e1):
                                self.out_windows[curr_id].setdefault(spk, []).append(list(overlap))

                        if spk not in matched:
                            continue

                        for s2, e2 in self.windows[key_2][matched[spk]]:
                            curr_iou = self.compute_iou(s1, e1, s2, e2)
                            # curr_iou = self.compute_overlap_ratio(s1, e1, s2, e2)
                            iou_values.setdefault(curr_id, []).append(curr_iou)
                            if curr_iou > self.iou_threshold: 
                                overlap = max(s1, s2), min(e1, e2)
                                # overlap = min(s1, s2), max(e1, e2)
                                # overlap = round(1/2*(s1+s2),1), round(1/2*(e1+e2),1)
                                self.out_windows[curr_id].setdefault(spk, []).append(list(overlap))


            #handle last window
            for spk in self.windows[key_2]:
                for s1, e1 in self.windows[key_2][spk]:
                    non_ovl_start, non_ovl_end = segments_non_ovl_intervals[curr_id][-1][0], segments_non_ovl_intervals[curr_id][-1][1]
                    if overlap := self.get_overlap_interval(non_ovl_start, non_ovl_end, s1, e1):
                        self.out_windows[curr_id].setdefault(spk, []).append(list(overlap))


    def merge_segments(self):
        for curr_id in self.out_windows:
            for spk, intervals in self.out_windows[curr_id].items():
                intervals.sort()
                merged = [intervals[0]]
                for start, end in intervals[1:]:
                    if end-start<self.threshold:
                        continue
                    if start <= merged[-1][1] + self.merge_tolerance:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                self.out_windows[curr_id][spk] = merged

    def run(self):
        self.stitch_segments()
        self.merge_segments()
        return self.out_windows

