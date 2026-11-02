# Benchmark results

Exact match (the whole tool call is correct) per slice.

## dev (n=121)

| system | standard | colloquial | asr | overall EM | intent acc | slot F1 |
|---|---|---|---|---|---|---|
| naive baseline (English-style slots) | 9.3% | 0.0% | 8.0% | **5.8%** | 100.0% | 30.4% |
| Tulkki − name inflection (nominative only) | 63.0% | 52.4% | 60.0% | **58.7%** | 100.0% | 78.5% |
| Tulkki − colloquial forms | 100.0% | 76.2% | 100.0% | **91.7%** | 100.0% | 96.1% |
| Tulkki − open-vocab & repair | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |
| **Tulkki (full)** | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |

Tulkki latency: 0.017 ms/utterance (pure Python, single core)

## heldout (n=85)

| system | standard | colloquial | asr | overall EM | intent acc | slot F1 |
|---|---|---|---|---|---|---|
| naive baseline (English-style slots) | 5.7% | 0.0% | 15.8% | **5.9%** | 100.0% | 33.1% |
| Tulkki − name inflection (nominative only) | 65.7% | 45.2% | 63.2% | **57.6%** | 100.0% | 76.4% |
| Tulkki − colloquial forms | 100.0% | 67.7% | 100.0% | **88.2%** | 100.0% | 94.8% |
| Tulkki − open-vocab & repair | 94.3% | 96.8% | 100.0% | **96.5%** | 100.0% | 97.7% |
| **Tulkki (full)** | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |

Tulkki latency: 0.019 ms/utterance (pure Python, single core)

## test2 (n=68)

| system | standard | colloquial | asr | overall EM | intent acc | slot F1 |
|---|---|---|---|---|---|---|
| naive baseline (English-style slots) | 0.0% | 0.0% | 15.4% | **2.9%** | 100.0% | 30.9% |
| Tulkki − name inflection (nominative only) | 50.0% | 56.0% | 53.8% | **52.9%** | 100.0% | 75.2% |
| Tulkki − colloquial forms | 90.0% | 76.0% | 100.0% | **86.8%** | 100.0% | 93.2% |
| Tulkki − open-vocab & repair | 73.3% | 96.0% | 100.0% | **86.8%** | 100.0% | 92.7% |
| **Tulkki (full)** | 90.0% | 96.0% | 100.0% | **94.1%** | 100.0% | 96.1% |

Tulkki latency: 0.036 ms/utterance (pure Python, single core)

## synthetic (n=3000)

| system | standard | colloquial | asr | overall EM | intent acc | slot F1 |
|---|---|---|---|---|---|---|
| naive baseline (English-style slots) | 2.2% | 2.7% | 2.0% | **2.3%** | 100.0% | 34.2% |
| Tulkki − name inflection (nominative only) | 51.0% | 49.4% | 48.3% | **49.7%** | 100.0% | 74.7% |
| Tulkki − colloquial forms | 100.0% | 70.8% | 84.0% | **85.4%** | 100.0% | 93.8% |
| Tulkki − open-vocab & repair | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |
| **Tulkki (full)** | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |

Tulkki latency: 0.020 ms/utterance (pure Python, single core)
