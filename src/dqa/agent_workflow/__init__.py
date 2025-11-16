# Based on: https://arize.com/docs/phoenix/integrations/python/pydantic/pydantic-tracing
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.pydantic_ai.utils import is_openinference_span
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import register
from dqa import EnvVars

# Setup Arize Phoenix OpenTelemetry tracing
# Add the OpenInference span processor
endpoint = f"{EnvVars.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces"
# Set up the tracer provider
tracer_provider = register(
    endpoint=endpoint,
    project_name="DQA",
    batch=True,
    protocol="http/protobuf",
    set_global_tracer_provider=False,
    auto_instrument=True,
)
exporter = OTLPSpanExporter(endpoint=endpoint, compression=Compression.Gzip)
tracer_provider.add_span_processor(
    OpenInferenceSpanProcessor(span_filter=is_openinference_span)
)
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

trace.set_tracer_provider(tracer_provider)
