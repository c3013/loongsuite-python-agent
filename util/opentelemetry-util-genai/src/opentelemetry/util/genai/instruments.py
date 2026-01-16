from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]

_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]

# Buckets for timing metrics (seconds) - for streaming token timing
_GEN_AI_CLIENT_TOKEN_TIMING_BUCKETS = [
    0.001,  # 1ms
    0.002,
    0.005,
    0.01,   # 10ms
    0.02,
    0.05,
    0.1,    # 100ms
    0.2,
    0.5,
    1.0,    # 1s
    2.0,
    5.0,
    10.0,   # 10s
    30.0,   # 30s
]


def create_duration_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
        description="Duration of GenAI client operation",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    )


def create_token_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
        description="Number of input and output tokens used by GenAI clients",
        unit="{token}",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    )


def create_time_to_first_token_histogram(meter: Meter) -> Histogram:
    """Create histogram for time to first token metric."""
    return meter.create_histogram(
        name="gen_ai.client.time_to_first_token",
        description="Time to first token for streaming responses",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_TIMING_BUCKETS,
    )


def create_time_per_output_token_histogram(meter: Meter) -> Histogram:
    """Create histogram for average time per output token metric."""
    return meter.create_histogram(
        name="gen_ai.client.time_per_output_token",
        description="Average time per output token",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_TIMING_BUCKETS,
    )


def create_time_between_token_histogram(meter: Meter) -> Histogram:
    """Create histogram for time between consecutive tokens metric."""
    return meter.create_histogram(
        name="gen_ai.client.time_between_token",
        description="Average time between consecutive tokens",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_TIMING_BUCKETS,
    )


def create_cached_tokens_histogram(meter: Meter) -> Histogram:
    """Create histogram for cached tokens metric."""
    return meter.create_histogram(
        name="gen_ai.usage.prompt_tokens_details.cached_tokens",
        description="Number of cached tokens from prompt",
        unit="{token}",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    )

