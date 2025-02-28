import logging
from typing import Dict, List, Tuple
import dspy
import litellm

from dspy.adapters.json_adapter import (
    _get_structured_outputs_response_format,
    parse_value,
    json_repair,
)

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX_DEEPSEEK = "deepseek"
LM_OUTPUT_KEY_REASONING = "reasoning"


class DeepseekJSONAdapter(dspy.JSONAdapter):
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        inputs = super(DeepseekJSONAdapter, self).format(signature, demos, inputs)
        inputs = (
            dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)
        )
        litellm.enable_json_schema_validation = True

        try:
            provider = lm.model.split("/", 1)[0] or "openai"
            params = litellm.get_supported_openai_params(
                model=lm.model, custom_llm_provider=provider
            )
            if params and "response_format" in params:
                try:
                    response_format = _get_structured_outputs_response_format(signature)
                    outputs = lm(
                        **inputs,
                        **lm_kwargs,
                        response_format=response_format,
                        # stream=True if "stream" in params else False,
                    )
                except Exception:
                    logger.debug(
                        "Failed to obtain response using signature-based structured outputs"
                        " response format: Falling back to default 'json_object' response format."
                        " Exception: {e}"
                    )
                    outputs = lm(
                        **inputs,
                        **lm_kwargs,
                        response_format={"type": "json_object"},
                        # stream=True if "stream" in params else False
                    )
            else:
                outputs = lm(**inputs, **lm_kwargs)

        except litellm.UnsupportedParamsError:
            outputs = lm(**inputs, **lm_kwargs)

        values = []

        for output in outputs:
            thinking_start_splits = output.split("<think>")
            thinking_end_splits = output.split("</think>")
            reasoning = (
                thinking_start_splits[1].split("</think>")[0].strip()
                if len(thinking_start_splits) > 1
                else output
            )
            response = (
                thinking_end_splits[1].strip()
                if len(thinking_end_splits) > 1
                else output
            )
            if LM_OUTPUT_KEY_REASONING in signature.output_fields.keys():
                value = self.parse(signature, response)
                # Remove reasoning because we will add it from the <think>...</think> content from Deepseek output
                value.pop(LM_OUTPUT_KEY_REASONING, None)
                if not value:
                    # Nothing parsed, so we just return the response as is
                    set_of_fields_excluding_reasoning = set(
                        signature.output_fields.keys()
                    ).difference([LM_OUTPUT_KEY_REASONING])
                    if len(set_of_fields_excluding_reasoning) == 1:
                        value[next(iter(set_of_fields_excluding_reasoning))] = response
                value[LM_OUTPUT_KEY_REASONING] = reasoning
            else:
                value = super(DeepseekJSONAdapter, self).parse(signature, output)

            values.append(value)

        return values

    def parse(self, signature, completion):
        if not completion:
            return {}
        fields = json_repair.loads(completion)
        if (
            not isinstance(fields, Dict)
            and not isinstance(fields, List)
            and not isinstance(fields, Tuple)
        ):
            # If the completion not a dictionary, a list or a tuple, do not try to parse it as a JSON dictionary
            return {}
        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # attempt to cast each value to type signature.output_fields[k].annotation
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        return fields
