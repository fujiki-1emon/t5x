import functools

from seqio import Feature
from seqio import vocabularies
import tensorflow as tf
from t5.data import preprocessors
from t5.data.utils import DEFAULT_EXTRA_IDS
from t5.data.dataset_providers import TaskRegistry
from t5.data.dataset_providers import TextLineTask

from sacrebleu import corpus_bleu


def bleu(targets, predictions):
  predictions = [tf.compat.as_text(x) for x in predictions]
  if isinstance(targets[0], list):
    targets = [[tf.compat.as_text(x) for x in target] for target in targets]
  else:
    targets = [tf.compat.as_text(x) for x in targets]
    targets = [targets]

  bleu_score = corpus_bleu(
  predictions, targets,
 smooth_method="exp",
 smooth_value=0.0,
 force=False,
 lowercase=False,
 tokenize="ja-mecab",
 use_effective_order=False)
  return {"bleu": bleu_score.score}


task_name = "snow_t15_23"

tsv_path = {
    "train": "gs://large_language_models_ja/datasets/snow_t15_23/snow_t15_23_train.tsv",
    "validation": "gs://large_language_models_ja/datasets/snow_t15_23/snow_t15_23_dev.tsv",
    "test": "gs://large_language_models_ja/datasets/snow_t15_23/snow_t15_23_test.tsv",
}

TaskRegistry.add(
    task_name,
    TextLineTask,
    split_to_filepattern=tsv_path,
    text_preprocessor=[
      functools.partial(preprocessors.parse_tsv, field_names=["inputs", "targets"]),
    ],
    output_features = Feature(vocabularies.SentencePieceVocabulary("gs://large_language_models_ja/pre_training/spiece.model", DEFAULT_EXTRA_IDS)),
    metric_fns=[bleu]
)
