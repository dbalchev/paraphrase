import numpy as np

class SciKitClassifier:
    def __init__(self, train_database, feature_gen, classifier):
        self._classifier = classifier
        self._feature_gen = feature_gen
        inputs = []
        outputs = []
        for data_row in train_database:
            if data_row.is_debatable():
                continue
            inputs.append(np.array(feature_gen(data_row)))
            outputs.append(1 if data_row.is_paraphrase() else 0)
        X, Y = np.array(inputs, dtype=np.float32), np.array(outputs)
#        print(X.shape, Y.shape, outp)
        self._classifier.fit(X, Y)

    def classify(self, data_row):
        features = self._feature_gen(data_row)
        p = self._classifier.predict(np.array([features], dtype=np.float32))
        return p[0] > 0.5
