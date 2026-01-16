# -*- coding: utf-8 -*-
"""Tests for LLM-specific metrics instrumentation."""

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder


class TestLLMMetrics:
    """Test LLM-specific metrics are recorded correctly."""

    @pytest.fixture
    def meter_provider(self):
        """Create a meter provider with in-memory metric reader."""
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        return provider, reader
    
    @pytest.fixture
    def tracer_provider(self):
        """Create a tracer provider."""
        return TracerProvider()

    def test_time_to_first_token_metric(self, meter_provider, tracer_provider):
        """Test that time_to_first_token metric is recorded."""
        provider, reader = meter_provider
        meter = provider.get_meter(__name__)
        recorder = InvocationMetricsRecorder(meter)

        # Create invocation with time_to_first_token
        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            time_to_first_token_s=0.15,
            monotonic_start_s=0.0,
        )

        # Create a mock span (we need a span to record metrics)
        tracer = tracer_provider.get_tracer(__name__)
        
        with tracer.start_as_current_span("test") as span:
            recorder.record(span, invocation)

        # Get metrics
        metrics_data = reader.get_metrics_data()
        
        # Find the time_to_first_token metric
        time_to_first_token_found = False
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "gen_ai.client.time_to_first_token":
                        time_to_first_token_found = True
                        assert len(metric.data.data_points) > 0
                        # Verify the value
                        for dp in metric.data.data_points:
                            assert dp.sum == 0.15
        
        assert time_to_first_token_found, "time_to_first_token metric not found"

    def test_time_per_output_token_metric(self, meter_provider, tracer_provider):
        """Test that time_per_output_token metric is recorded."""
        provider, reader = meter_provider
        meter = provider.get_meter(__name__)
        recorder = InvocationMetricsRecorder(meter)

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            time_per_output_token_s=0.025,
            monotonic_start_s=0.0,
        )

        tracer = tracer_provider.get_tracer(__name__)
        
        with tracer.start_as_current_span("test") as span:
            recorder.record(span, invocation)

        metrics_data = reader.get_metrics_data()
        
        time_per_output_token_found = False
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "gen_ai.client.time_per_output_token":
                        time_per_output_token_found = True
                        assert len(metric.data.data_points) > 0
                        for dp in metric.data.data_points:
                            assert dp.sum == 0.025
        
        assert time_per_output_token_found, "time_per_output_token metric not found"

    def test_time_between_token_metric(self, meter_provider, tracer_provider):
        """Test that time_between_token metric is recorded."""
        provider, reader = meter_provider
        meter = provider.get_meter(__name__)
        recorder = InvocationMetricsRecorder(meter)

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            time_between_token_s=0.018,
            monotonic_start_s=0.0,
        )

        tracer = tracer_provider.get_tracer(__name__)
        
        with tracer.start_as_current_span("test") as span:
            recorder.record(span, invocation)

        metrics_data = reader.get_metrics_data()
        
        time_between_token_found = False
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "gen_ai.client.time_between_token":
                        time_between_token_found = True
                        assert len(metric.data.data_points) > 0
                        for dp in metric.data.data_points:
                            assert dp.sum == 0.018
        
        assert time_between_token_found, "time_between_token metric not found"

    def test_cached_tokens_metric(self, meter_provider, tracer_provider):
        """Test that cached_tokens metric is recorded."""
        provider, reader = meter_provider
        meter = provider.get_meter(__name__)
        recorder = InvocationMetricsRecorder(meter)

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            cached_tokens=512,
            monotonic_start_s=0.0,
        )

        tracer = tracer_provider.get_tracer(__name__)
        
        with tracer.start_as_current_span("test") as span:
            recorder.record(span, invocation)

        metrics_data = reader.get_metrics_data()
        
        cached_tokens_found = False
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "gen_ai.usage.prompt_tokens_details.cached_tokens":
                        cached_tokens_found = True
                        assert len(metric.data.data_points) > 0
                        for dp in metric.data.data_points:
                            assert dp.sum == 512
        
        assert cached_tokens_found, "cached_tokens metric not found"

    def test_all_new_metrics_together(self, meter_provider, tracer_provider):
        """Test that all new metrics can be recorded together."""
        provider, reader = meter_provider
        meter = provider.get_meter(__name__)
        recorder = InvocationMetricsRecorder(meter)

        invocation = LLMInvocation(
            request_model="test-model",
            provider="test-provider",
            time_to_first_token_s=0.15,
            time_per_output_token_s=0.025,
            time_between_token_s=0.018,
            cached_tokens=512,
            input_tokens=100,
            output_tokens=50,
            monotonic_start_s=0.0,
        )

        tracer = tracer_provider.get_tracer(__name__)
        
        with tracer.start_as_current_span("test") as span:
            recorder.record(span, invocation)

        metrics_data = reader.get_metrics_data()
        
        # Collect all metric names
        metric_names = set()
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    metric_names.add(metric.name)
        
        # Verify all new metrics are present
        expected_metrics = {
            "gen_ai.client.time_to_first_token",
            "gen_ai.client.time_per_output_token",
            "gen_ai.client.time_between_token",
            "gen_ai.usage.prompt_tokens_details.cached_tokens",
            "gen_ai.client.operation.duration",  # existing metric
            "gen_ai.client.token.usage",  # existing metric
        }
        
        for expected_metric in expected_metrics:
            assert expected_metric in metric_names, f"{expected_metric} not found in metrics"
