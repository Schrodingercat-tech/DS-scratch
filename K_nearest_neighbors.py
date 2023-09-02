from linalg import linalg,vector
from typing import NamedTuple,List,Counter

class LabeledPoint(NamedTuple):
    feature_vector : vector
    label_name :str

class Models:
    @staticmethod
    def majority_label(labels:List[str])->str:
        label_counts = Counter(labels)
        maxcountLabel,maxcount = label_counts.most_common(1)[0]
        # note other labels may also have the same count so we will se no of winner who have same count
        no_of_same_maxcountLabels = len([count for count in label_counts.values() if count == maxcount])
        if no_of_same_maxcountLabels == 1: return maxcountLabel
        else : return Models.majority_label(labels[:-1])
    @staticmethod
    def Knn_classifer(
        k : int , # no of neibouring points default 3,
        data :List[LabeledPoint],
        new_point :vector)->str:
        sort_by_distance = sorted(data,key= lambda lp:linalg.distance(lp.feature_vector-new_point))
        k_nearest_labels = [lp.label_name for lp in sort_by_distance[:k]]
        return Models.majority_label(k_nearest_labels)

