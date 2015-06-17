from .models import *
from .evaluate import *
from .features import concat
from .ngram import C1, C2, V1, V2
from .skclassifier import SciKitClassifier
from sklearn.pipeline import make_pipeline
from sklearn import svm
import sklearn.preprocessing as pre


if __name__ == "__main__":
    train_database = read_database("train.data", TrainDataRow)
    test_database  = read_database("test.data", TestDataRow)
    test_labels    = read_test_labels("test.label")
    features = [C1, C2, V1, V2] + \
        [concat(c, v) for c in [C1, C2] for v in [V1, V2]]

    for features_gen in features:
        svm_pipeline = \
            make_pipeline(pre.StandardScaler(), svm.SVC(class_weight="auto"))
        classifier = SciKitClassifier(train_database, features_gen, svm_pipeline)
        print("features", features_gen.name)
        print(evaluate(classifier, test_database, test_labels))

    # print("\n".join(str(x) for x in train_database[0:10]))
