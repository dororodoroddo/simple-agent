import inspect
import copy
import json
import re
import sys
sys.set_int_max_str_digits(0)  # Exact caller-provided numbers; no application digit cap.
from dataclasses import dataclass
from decimal import Decimal, localcontext
from .errors import RunError


def compile_schema(schema):
    """JSON Schema draft 7; no defaults, coercion or network reference resolution."""
    import fastjsonschema
    def reject_remote(value):
        if isinstance(value, dict):
            ref = value.get('$ref')
            if isinstance(ref, str) and not ref.startswith('#'):
                raise ValueError('Use a local #/$defs or #/definitions reference; schemas never fetch URLs.')
            for child in value.values(): reject_remote(child)
        elif isinstance(value, list):
            for child in value: reject_remote(child)
    reject_remote(schema)
    return fastjsonschema.compile(schema, use_default=False)


def validate_schema(value, schema, code="invalid_tool_arguments"):
    import fastjsonschema
    try:
        compile_schema(schema)(value)
    except fastjsonschema.JsonSchemaException as exc:
        raise RunError(code, str(exc), details={"path": list(exc.path), "rule": exc.rule}) from exc


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


NUMBER = {"type": "string", "pattern": r"^-?\d+(?:\.\d+)?$"}
CALCULATE_INPUT = object_schema(
    {
        "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
        },
        "a": NUMBER,
        "b": NUMBER,
    }
)


def calculate(arguments):
    validate_schema(arguments, CALCULATE_INPUT)
    a, b = Decimal(arguments["a"]), Decimal(arguments["b"])
    from fractions import Fraction
    operations = {"add": lambda: Fraction(a) + Fraction(b), "subtract": lambda: Fraction(a) - Fraction(b),
                  "multiply": lambda: Fraction(a) * Fraction(b), "divide": lambda: Fraction(a) / Fraction(b)}
    try:
        exact = operations[arguments["operation"]]()
    except ZeroDivisionError as exc:
        raise RunError("invalid_tool_arguments", "0으로 나눌 수 없습니다.") from exc
    # Fractions preserve every input digit and do not silently round recurring decimals.
    result = str(exact.numerator) if exact.denominator == 1 else f"{exact.numerator}/{exact.denominator}"
    operation = arguments["operation"]
    return {
        "operation": operation,
        "a": str(a),
        "b": str(b),
        "result": str(result),
        "status": "success",
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    handler: object
    permission: str
    read_only: bool = True
    validate_input: object = None
    validate_output: object = None
    external: bool = False  # Leaf external I/O; composite functions use registered leaf adapters.

    def model_schema(self, *, strict=None):
        if strict is True:
            from .function_calling import validate_strict_schema
            validate_strict_schema(self.input_schema)
        schema = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
        if strict is not None: schema['strict'] = strict
        return copy.deepcopy(schema)


class ToolRegistry:
    def __init__(self, specs=()):
        self.specs = {}
        self.validators = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec):
        if spec.name in self.specs:
            raise ValueError("duplicate tool")
        self.validators[spec.name] = (compile_schema(spec.input_schema), compile_schema(spec.output_schema))
        self.specs[spec.name] = spec

    def get(self, name):
        if not isinstance(name, str) or name not in self.specs:
            raise RunError(
                "tool_not_allowed", "허용되지 않은 도구입니다.", details={"tool": name}
            )
        return self.specs[name]

    def schemas(self, permissions=None):
        return [
            s.model_schema()
            for s in self.specs.values()
            if permissions is None or s.permission in permissions
        ]

    def authorize(self, spec, context):
        if spec.permission not in context.permissions:
            raise RunError("tool_permission_denied", "도구 실행 권한이 없습니다.")

    def validate_arguments(self, spec, args, context):
        import fastjsonschema
        try: self.validators[spec.name][0](args)
        except fastjsonschema.JsonSchemaException as exc:
            raise RunError("invalid_tool_arguments", str(exc), details={"path": list(exc.path)}) from exc
        if spec.validate_input:
            spec.validate_input(args, context)

    async def invoke(self, spec, args, context):
        result = spec.handler(args, context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def validate_result(self, spec, result, context):
        import fastjsonschema
        try: self.validators[spec.name][1](result)
        except fastjsonschema.JsonSchemaException as exc:
            raise RunError("invalid_tool_output", str(exc), details={"path": list(exc.path)}) from exc
        if spec.validate_output:
            spec.validate_output(result, context)


def default_registry():
    string = {"type": "string"}
    registry = ToolRegistry(
        [
            ToolSpec(
                "calculate",
                "제공된 숫자의 정확한 사칙연산을 수행합니다.",
                CALCULATE_INPUT,
                object_schema(
                    {
                        "operation": CALCULATE_INPUT["properties"]["operation"],
                        "a": string,
                        "b": string,
                        "result": string,
                        "status": {"type": "string", "enum": ["success"]},
                    }
                ),
                lambda args, ctx: calculate(args),
                "calculate",
            ),
        ]
    )

    from .providers import LLM_CONFIG_SCHEMA, validate_llm_config
    async def llm(args, ctx):
        return await ctx.services._call_llm(args["prompt"], args["files"], config=args["config"])
    registry.register(ToolSpec(
        "callLLM", "호출자가 지정한 설정과 텍스트·혼합 파일로 생성 요청을 한 번 시도합니다.",
        object_schema({"prompt": string, "files": {"type": "array", "items": string}, "config": LLM_CONFIG_SCHEMA}),
        {"type": "object"}, llm, "llm", validate_input=lambda args, ctx: validate_llm_config(args["config"])))

    async def save_data(args, ctx):
        return await ctx.store.save_data(args["scope"], args["key"], args["value"])

    async def load_data(args, ctx):
        return await ctx.store.load_data(args["scope"], args["key"])

    name = {"type": "string", "minLength": 1}
    artifact = {"type": "object", "properties": {
        "id": name, "filename": name, "mime": name,
        "bytes": {"type": "integer", "minimum": 0},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "runId": {"type": ["string", "null"]}},
        "required": ["id", "filename", "mime", "bytes", "sha256", "runId"]}

    async def file(args, ctx):
        return await ctx.services._file(args["ref"])

    async def file_metadata(args, ctx):
        return await ctx.services._file_metadata(args['ref'])

    registry.register(ToolSpec('file_metadata', '본문을 받지 않고 저장된 메타데이터 또는 HEAD 헤더를 조회합니다.',
        object_schema({'ref': name}), {'type': 'object'}, file_metadata, 'file_read'))

    file_result = {**artifact, "properties": {**artifact["properties"],
        "format": object_schema({
            "kind": {"enum": ["document", "image", "audio", "video", "archive", "unknown"]},
            "extension": string, "mime": name,
            "status": {"enum": ["identified", "metadata_only", "unknown", "conflict"]},
            "evidence": object_schema({"filenameExtension": string, "declaredMime": string,
                                       "signature": {"type": ["string", "null"]}}),
            "issues": {"type": "array", "items": {"enum": ["header_mismatch", "header_missing", "mime_mismatch"]}}}),
        "modelInputs": object_schema({api: {"type": "boolean"} for api in
                                      ("openai/responses", "openai/chat/completions", "gemini/interactions")})},
        "required": [*artifact["required"], "format", "modelInputs"]}
    registry.register(ToolSpec(
        "file", "명시한 원본을 읽어 해시·형식 근거·충돌과 현재 API 어댑터의 입력 형식 지원 여부를 반환합니다. 내용이나 모델별 지원은 검증하지 않습니다.",
        object_schema({"ref": name}), file_result, file, "file_read"))


    registry.register(ToolSpec(
        "save_data", "지정한 범위와 키에 JSON 데이터를 저장합니다.",
        object_schema({"scope": name, "key": name, "value": {}}),
        object_schema({"scope": name, "key": name, "saved": {"const": True}}),
        save_data, "data_write", read_only=False, external=True))
    registry.register(ToolSpec(
        "load_data", "지정한 범위와 키에 저장된 JSON 데이터를 조회합니다.",
        object_schema({"scope": name, "key": name}),
        object_schema({"scope": name, "key": name, "found": {"type": "boolean"}, "value": {}}),
        load_data, "data_read", external=True))


    from .rag import search, valid_space_vector
    vector = {"type": "array", "items": {"type": "number"}, "minItems": 1}

    async def embed(args, ctx):
        return await ctx.services._embed_text(args["text"], args.get("dimensions"))

    async def rag_put(args, ctx):
        return await ctx.store.rag_put(args["scope"], args["key"],
            {"key": args["key"], "text": args["text"], "vector": args["vector"],
             "space": args["space"], "metadata": args["metadata"]})

    async def rag_search(args, ctx):
        return await search(ctx.store, args["scope"], args["space"], args["query"], args["vector"], args["limit"])

    registry.register(ToolSpec(
        "embed_text", "text-embedding-3-small로 텍스트의 임베딩 벡터를 생성합니다.",
        {"type": "object", "properties": {"text": {"type": "string", "minLength": 1},
          "dimensions": {"type": "integer", "minimum": 1, "maximum": 1536}},
         "required": ["text"], "additionalProperties": False},
        {"type": "object"}, embed, "embedding"))
    registry.register(ToolSpec(
        "rag_put", "텍스트·출처 정보와 임베딩을 지정한 검색 범위에 저장합니다.",
        object_schema({"scope": name, "key": name, "text": string, "space": name,
                       "vector": vector, "metadata": {"type": "object"}}),
        object_schema({"scope": name, "key": name, "saved": {"const": True}}),
        rag_put, "rag_write", read_only=False, external=True,
        validate_input=lambda args, ctx: valid_space_vector(args["space"], args["vector"])))
    registry.register(ToolSpec(
        "rag_search", "같은 임베딩 공간에서 키워드와 벡터를 결합해 자료를 조회합니다.",
        object_schema({"scope": name, "space": name, "query": {"type": "string", "minLength": 1}, "vector": vector,
                       "limit": {"type": "integer", "minimum": 1}}),
        object_schema({"scope": name, "space": name, "matches": {"type": "array"}}),
        rag_search, "rag_read", external=True,
        validate_input=lambda args, ctx: valid_space_vector(args["space"], args["vector"])))
    from .pdr import register_tools
    register_tools(registry)
    from .planner import planner_nodes
    for node in planner_nodes(): registry.register(node)
    return registry
