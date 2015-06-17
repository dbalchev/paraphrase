from itertools import chain
from .names import with_name
def concat(*sub_features):
    def feature_gen(data_row):
        return list(chain.from_iterable(sf(data_row) for sf in sub_features))
    return with_name(
        feature_gen,
        "[" + ",".join(sf.name for sf in sub_features) + "]")
