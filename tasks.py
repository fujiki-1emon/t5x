import functools

import seqio
from t5.data import preprocessors
from t5.data.utils import DEFAULT_EXTRA_IDS
from t5.evaluation import metrics

from ul2_objective import ul2_objective


TaskRegistry = seqio.TaskRegistry

sentencepiece_model_file = "gs://large_language_models_ja_v3/spiece.model"
vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file, DEFAULT_EXTRA_IDS)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=vocab, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=vocab, add_eos=True)
}

# values from UL2 paper https://arxiv.org/pdf/2205.05131.pdf chapter 3.1.2 table 1
R_DENOISER_SPAN_LENGTHS = [3.0, 8.0]
X_DENOISER_SPAN_LENGTHS = [3.0, 8.0, 64.0, 64.0]

R_DENOISER_CORRUPT_RATES = [0.15, 0.15]
X_DENOISER_CORRUPT_RATES = [0.5, 0.5, 0.15, 0.5]

R_DENOISER_TOKEN_PREFIX = '[NLU]'
X_DENOISER_TOKEN_PREFIX = '[NLG]'
S_DENOISER_TOKEN_PREFIX = '[S2S]'


TaskRegistry.add(
   "pre_training.obpc.ul2_objective",
    source=seqio.TfdsDataSource(tfds_name="obpc:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(
            ul2_objective,
            # shard_ds=False,
            use_prefix_lm_task=True,   
            rates=[0.4 / len(R_DENOISER_SPAN_LENGTHS)] * len(R_DENOISER_SPAN_LENGTHS) \
                + [0.4 / len(X_DENOISER_SPAN_LENGTHS)] * len(X_DENOISER_SPAN_LENGTHS) \
                + [0.2],
            mean_noise_span_lengths=R_DENOISER_SPAN_LENGTHS + X_DENOISER_SPAN_LENGTHS,
            noise_densities=R_DENOISER_CORRUPT_RATES + X_DENOISER_CORRUPT_RATES,
            optional_task_prefixes=[R_DENOISER_TOKEN_PREFIX] * len(R_DENOISER_SPAN_LENGTHS) \
                + [X_DENOISER_TOKEN_PREFIX] * len(X_DENOISER_SPAN_LENGTHS) \
                + [S_DENOISER_TOKEN_PREFIX],
            reserved_for_packing=True,
        ),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={
        "inputs": DEFAULT_OUTPUT_FEATURES["inputs"],
        "targets": DEFAULT_OUTPUT_FEATURES["targets"],
    },
    metric_fns=[metrics.accuracy]
)


TaskRegistry.add(
   "pre_training.ja_large_publics.ul2_objective",
    source=seqio.TfdsDataSource(tfds_name="ja_large_publics:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        functools.partial(
            ul2_objective,
            # shard_ds=False,
            use_prefix_lm_task=True,
            rates=[0.4 / len(R_DENOISER_SPAN_LENGTHS)] * len(R_DENOISER_SPAN_LENGTHS) \
                + [0.4 / len(X_DENOISER_SPAN_LENGTHS)] * len(X_DENOISER_SPAN_LENGTHS) \
                + [0.2],
            mean_noise_span_lengths=R_DENOISER_SPAN_LENGTHS + X_DENOISER_SPAN_LENGTHS,
            noise_densities=R_DENOISER_CORRUPT_RATES + X_DENOISER_CORRUPT_RATES,
            optional_task_prefixes=[R_DENOISER_TOKEN_PREFIX] * len(R_DENOISER_SPAN_LENGTHS) \
                + [X_DENOISER_TOKEN_PREFIX] * len(X_DENOISER_SPAN_LENGTHS) \
                + [S_DENOISER_TOKEN_PREFIX],
            reserved_for_packing=True,
        ),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={
        "inputs": DEFAULT_OUTPUT_FEATURES["inputs"],
        "targets": DEFAULT_OUTPUT_FEATURES["targets"],
    },
    metric_fns=[metrics.accuracy]
)