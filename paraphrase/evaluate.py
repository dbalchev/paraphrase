
class Score:
    def __init__(self, guess_answer_pairs):
        tp, tn, fp, fn = [0, 0, 0, 0]
        for guess, answer in guess_answer_pairs:
            if guess == True and answer == False:
                fp += 1.0
            elif guess == False and answer == True:
                fn += 1.0
            elif guess == True and answer == True:
                tp += 1.0
            elif guess == False and answer == False:
                tn += 1.0
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn)  if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall)
        self.true_positive = tp
        self.true_negative = tn
        self.false_positive = fp
        self.false_negative = fn
        self.precision = precision
        self.recall = recall
        self.f1 = f1
    def __str__(self):
        return "PRECISION: {:.4}, RECALL: {:.4}, F1: {:.4}" \
                .format(self.precision, self.recall, self.f1)

def evaluate(classifier, test_database, test_labels):
    return Score((classifier.classify(dr), ans) \
        for dr, ans in zip(test_database, test_labels) \
        if not dr.is_debatable())
