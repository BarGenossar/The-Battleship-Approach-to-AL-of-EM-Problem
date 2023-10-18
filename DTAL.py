import re
import math


class DTAL:
    def __init__(self, available_pool_path, k):
        self.available_pool_path = available_pool_path
        self.k = k
        self.d_plus, self.d_minus = self.create_confidences_available_pool()
        self.pos_budget = int(min(k / 2, len(self.d_plus)))
        self.neg_budget = k - self.pos_budget
        self.likely_false_positive, self.high_confidence_positive = self.top_bottom_k('pos')
        # set of samples' indices with match prediction with high and low entropy (respectively)
        self.likely_false_negative, self.high_confidence_negative = self.top_bottom_k('neg')
        # set of samples' indices with non-match prediction with high and low entropy (respectively)
        self.likely_false = set.union(self.likely_false_positive, self.likely_false_negative)

    def create_confidences_available_pool(self):
        """
        Create and return a mapping from row indices to confidence and entropy values, separately for match and
        non-match. Return the mapping.
        """
        d_plus, d_minus = dict(), dict()
        preds_file = open(self.available_pool_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            confidence = float(re.sub("[^0-9.]", "", line.split("match_confidence")[1].split("pooler")[0]))
            prediction = int(re.sub("[^0-9]", "", line.split("\"match\"")[1][3]))
            p = abs(1 - prediction - confidence)
            h = self.calc_entropy(p)
            (d_plus if prediction else d_minus)[id_val] = h
        preds_file.close()
        return d_plus, d_minus

    def top_bottom_k(self, type):
        """
        Get a dictionary maps from indices to (confidence, entropy) and return a lists with indices corresponds to 
        the top_k entropy values, and last_k entropy values. 
        """
        d_dict = self.d_plus if type == 'pos' else self.d_minus
        budget = self.pos_budget if type == 'pos' else self.neg_budget
        d_items = sorted(d_dict.items(), key=lambda item: item[1], reverse=True)
        likely_false, high_confidence = set(), set()
        for i in range(budget):
            try:
                likely_false.add(d_items[i][0])
                high_confidence.add(d_items[-(i + 1)][0])
            except:
                print(f'The number of relevant predicted samples of this label is too small')
                print(f'K={i+1}')
                break
        return likely_false, high_confidence

    @staticmethod
    def calc_entropy(p):
        try:
            return -p * math.log(p) - (1 - p) * math.log(1 - p)
        except:
            return 0
