# Copyright The OpenTelemetry Authors
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

"""
Extended metrics recorder for GenAI invocations.

This module provides LoongSuite GenAI metrics recording following ARMS semantic conventions.
It supports multiple GenAI invocation types: chat, generate_content, embedding, execute_tool, invoke_agent,
create_agent, retrieve, and rerank.

This is just an empty implementation for now, which is a placeholder for enterprise implementation.
"""

from __future__ import annotations

import logging
import timeit
from numbers import Number
from typing import Dict, Optional, Union

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.util.genai._extended_memory.memory_types import (
    MemoryInvocation,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RerankInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.instruments import (
    create_agent_duration_histogram,
    create_tool_duration_histogram,
    create_workflow_duration_histogram,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.types import AttributeValue

_logger = logging.getLogger(__name__)


class ExtendedInvocationMetricsRecorder(InvocationMetricsRecorder):
    """
    Extended metrics recorder that supports multiple GenAI invocation types.

    This class provides LoongSuite GenAI metrics recording following ARMS semantic conventions.
    It supports:
    - Chat/Generate content operations
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Create agent operations
    - Retrieve documents operations
    - Rerank documents operations
    """

    def __init__(self, meter: Meter):
        """Initialize the extended metrics recorder with additional histograms."""
        super().__init__(meter)
        # Add extended histograms for agent, workflow, and tool duration
        self._agent_duration_histogram: Histogram = (
            create_agent_duration_histogram(meter)
        )
        self._workflow_duration_histogram: Histogram = (
            create_workflow_duration_histogram(meter)
        )
        self._tool_duration_histogram: Histogram = (
            create_tool_duration_histogram(meter)
        )

    def record_extended(
        self,
        span: Optional[Span],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrieveInvocation,
            RerankInvocation,
            MemoryInvocation,
        ],
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record duration and token metrics for any GenAI invocation type.

        This method automatically routes to the appropriate handler based on
        the invocation type.

        Args:
            span: The span associated with this invocation
            invocation: The invocation object (any supported type)
            error_type: Optional error type if the invocation failed
        """
        if isinstance(invocation, LLMInvocation):
            self.record(span, invocation, error_type=error_type)
            return
        
        if isinstance(invocation, InvokeAgentInvocation):
            self._record_agent_duration(span, invocation, error_type)
            return
        
        if isinstance(invocation, ExecuteToolInvocation):
            self._record_tool_duration(span, invocation, error_type)
            return
        
        # TODO: Implement other invocation types as needed

    def _record_agent_duration(
        self,
        span: Optional[Span],
        invocation: InvokeAgentInvocation,
        error_type: Optional[str] = None,
    ) -> None:
        """Record duration metrics for agent operations."""
        if span is None:
            return

        # Calculate duration
        duration_seconds: Optional[float] = None
        if invocation.monotonic_start_s is not None:
            duration_seconds = max(
                timeit.default_timer() - invocation.monotonic_start_s, 0.0
            )

        if (
            duration_seconds is not None
            and isinstance(duration_seconds, Number)
            and duration_seconds >= 0
        ):
            attributes: Dict[str, AttributeValue] = {}
            
            # Add gen_ai.operation.name attribute (always set for agent operations)
            attributes[GenAI.GEN_AI_OPERATION_NAME] = (
                GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
            )
            
            # Add error.type if present
            if error_type:
                attributes["error.type"] = error_type

            span_context = set_span_in_context(span)
            self._agent_duration_histogram.record(
                duration_seconds,
                attributes=attributes,
                context=span_context,
            )

    def _record_tool_duration(
        self,
        span: Optional[Span],
        invocation: ExecuteToolInvocation,
        error_type: Optional[str] = None,
    ) -> None:
        """Record duration metrics for tool operations."""
        if span is None:
            return

        # Calculate duration
        duration_seconds: Optional[float] = None
        if invocation.monotonic_start_s is not None:
            duration_seconds = max(
                timeit.default_timer() - invocation.monotonic_start_s, 0.0
            )

        if (
            duration_seconds is not None
            and isinstance(duration_seconds, Number)
            and duration_seconds >= 0
        ):
            attributes: Dict[str, AttributeValue] = {}
            
            # Add gen_ai.tool.name attribute
            if invocation.tool_name:
                attributes["gen_ai.tool.name"] = invocation.tool_name
            
            # Add error.type if present
            if error_type:
                attributes["error.type"] = error_type

            span_context = set_span_in_context(span)
            self._tool_duration_histogram.record(
                duration_seconds,
                attributes=attributes,
                context=span_context,
            )


__all__ = ["ExtendedInvocationMetricsRecorder"]
