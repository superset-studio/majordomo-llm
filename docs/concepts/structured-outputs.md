# Structured Outputs

## The Provider Fragmentation Problem

LLM providers implement structured outputs differently:

- **OpenAI** uses `response_format` with JSON schemas
- **Anthropic** uses tool calling to enforce structure
- **Gemini** uses `response_schema` in generation config
- **Others** rely on prompt engineering with varying reliability

This fragmentation creates problems:

1. **Vendor lock-in**: Code written for one provider's structured output API won't work with another
2. **Inconsistent validation**: Some providers validate server-side, others don't
3. **Schema translation**: Each provider has its own schema format and quirks
4. **Migration overhead**: Switching providers requires rewriting extraction logic

## One Method, All Providers

majordomo-llm provides two structured-output methods:

- `get_structured_json_response()` accepts a **Pydantic model** and returns a validated, typed Python object
- `get_json_schema_response()` accepts a **raw JSON Schema dict** and returns canonical JSON text
- Both methods work identically across **all supported providers**
- Both methods handle provider-specific implementation details internally

## Basic Usage

```python
from pydantic import BaseModel, Field
from majordomo_llm import get_llm_instance

class ExtractedData(BaseModel):
    title: str = Field(description="Document title")
    summary: str = Field(description="Brief summary")
    keywords: list[str] = Field(description="Key topics")

# Same code works with any provider
llm = get_llm_instance("anthropic", "claude-sonnet-5")
# llm = get_llm_instance("openai", "gpt-4.1")
# llm = get_llm_instance("gemini", "gemini-2.5-flash")

response = await llm.get_structured_json_response(
    response_model=ExtractedData,
    user_prompt="Extract info from: [your document text]",
)

# response.content is a validated ExtractedData instance
print(response.content.title)
print(response.content.keywords)
```

## Raw JSON Schema Usage

Use `get_json_schema_response()` when your schema comes from a file, registry, runtime
builder, or another system and you do not want to synthesize a Pydantic model:

```python
import json

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "keywords"],
}

response = await llm.get_json_schema_response(
    user_prompt="Extract info from: [your document text]",
    response_schema=schema,
    schema_name="ExtractedData",
)

# response.content is canonical JSON: sorted keys, no extra whitespace
data = json.loads(response.content)
```

## Under the Hood

1. Pydantic models are converted to JSON schema via `model_json_schema()` when needed
2. Raw schemas are translated to provider-specific format (tool definition for Anthropic, response format for OpenAI/Cohere/DeepSeek, response schema for Gemini)
3. Provider responses are parsed, lightly repaired when possible, and validated against the JSON schema
4. Raw-schema responses are serialized canonically; Pydantic responses are validated into typed objects

## Next Steps

See the [Structured Outputs recipe](../recipes/structured-outputs.md) for more examples including enums, nested models, and constrained fields.
