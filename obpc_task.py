import functools

from t5.data import preprocessors
from t5.data.utils import DEFAULT_EXTRA_IDS
import seqio
from seqio import TaskRegistry


sentencepiece_model_file = "gs://large_language_models_ja/pre_training/spiece.model"
vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file, DEFAULT_EXTRA_IDS)

TaskRegistry.add(
    "obpc_span_corruption",
    source=seqio.TfdsDataSource(tfds_name="obpc_train_0_9:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=vocab, add_eos=True, required=False), 
        "targets": seqio.Feature(vocabulary=vocab, add_eos=True),
    },
    metric_fns=[],
)
