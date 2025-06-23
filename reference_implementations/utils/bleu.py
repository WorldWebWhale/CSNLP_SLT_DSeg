# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


from tqdm import tqdm
import math, time, collections

def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order: int = 4,
                 smooth: bool = False,
                 desc: str = "⏳ BLEU"):
    """
    Computes corpus-level BLEU with an optional tqdm progress-bar.
    Args / Returns identical to the original function.
    """
    t0 = time.time()

    matches_by_order          = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = translation_length = 0

    # tqdm progress-bar
    for refs, hyp in tqdm(zip(reference_corpus, translation_corpus),
                          total=len(translation_corpus),
                          desc=desc,
                          unit="sent"):
        reference_length   += min(len(r) for r in refs)
        translation_length += len(hyp)

        merged_ref_ngram_counts = collections.Counter()
        for r in refs:
            merged_ref_ngram_counts |= _get_ngrams(r, max_order)
        hyp_ngram_counts = _get_ngrams(hyp, max_order)

        overlap = hyp_ngram_counts & merged_ref_ngram_counts
        for ngram, match_count in overlap.items():
            matches_by_order[len(ngram) - 1] += match_count

        for order in range(1, max_order + 1):
            possible = len(hyp) - order + 1
            if possible > 0:
                possible_matches_by_order[order - 1] += possible

    # --- identical to the original implementation --------------------------
    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.) / \
                            (possible_matches_by_order[i] + 1.)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]

    geo_mean = (math.exp(sum(math.log(p) / max_order for p in precisions))
                if min(precisions) > 0 else 0.0)

    ratio = translation_length / reference_length
    bp    = 1.0 if ratio > 1.0 else math.exp(1 - 1. / ratio)
    bleu  = geo_mean * bp
    # -----------------------------------------------------------------------

    tqdm.write(f"✅ BLEU computed in {time.time()-t0:.1f}s")
    return bleu, precisions, bp, ratio, translation_length, reference_length

