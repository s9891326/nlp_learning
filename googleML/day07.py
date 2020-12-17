# tf.Estimator API
import tempfile
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app

# load iris datasets
# train_data = tfds.load(name=datasets_name, split=[tfds.core.ReadInstruction('train')])
# ds, info = tfds.load('mnist', split='train', with_info=True)
# ds, info = tfds.load("iris", split='train', with_info=True)


# print(ds)
# print(info.features)

# print(ds.data)
# print(ds.label)
# tfds.as_dataframe(ds.take(4).cache().repeat(), info)
# fig = tfds.show_examples(ds, info)


# Set up a linear classifier.


def train_input_fn():
    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic = tf.data.experimental.make_csv_dataset(
        titanic_file, batch_size=32,
        label_name="survived")
    titanic_batches = (
        titanic.cache().repeat().shuffle(500)
            .prefetch(tf.data.experimental.AUTOTUNE))
    return titanic_batches

def main(_):
    age = tf.feature_column.numeric_column('age')
    cls = tf.feature_column.categorical_column_with_vocabulary_list('class', ['First', 'Second', 'Third'])
    embark = tf.feature_column.categorical_column_with_hash_bucket('embark_town', 32)

    model_dir = tempfile.mkdtemp()
    model = tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=[embark, cls, age],
        n_classes=2
    )
    model = model.train(input_fn=train_input_fn, steps=100)

    result = model.evaluate(train_input_fn, steps=10)

    for key, value in result.items():
        print(key, ":", value)

    for pred in model.predict(train_input_fn):
        for key, value in pred.items():
            print(key, ":", value)
        break


if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    app.run(main)
