# Theta-Augmented Gradient Boosting Embeddings for Tabular Similarity Research
# Based on Bar-Natan & van der Veen (2025) - arXiv:2509.18456
# "A Fast, Strong, Topologically Meaningful and Fun Knot Invariant"

__version__ = "0.2.0"

# Exposed public API
from .braid_word import row_to_braid_word, generate_braid_words
from .braid_utils import BraidClosure, braid_word_to_closure, validate_braid_words_batch, n_strands_from_feature_order
from .theta_eval import compute_theta_features, theta_eval
from .theta_exact import compute_exact_theta_features, alexander_polynomial_eval
from .similarity_search import run_full_similarity_benchmark
