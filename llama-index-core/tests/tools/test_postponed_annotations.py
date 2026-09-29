from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow import Context


class Item(BaseModel):
    name: str


def add_item(item: Item, count: Annotated[int, "How many to add"] = 1) -> str:
    return f"{count} x {item.name}"


async def add_item_with_ctx(ctx: Context, item: Item) -> str:
    return item.name


def test_schema_resolves_postponed_annotations() -> None:
    schema = create_schema_from_function("AddItem", add_item).model_json_schema()
    assert "Item" in schema["$defs"]
    assert schema["properties"]["count"]["description"] == "How many to add"


def test_context_detected_with_postponed_annotations() -> None:
    tool = FunctionTool.from_defaults(async_fn=add_item_with_ctx)
    assert tool.requires_context is True
    assert tool.ctx_param_name == "ctx"
    assert "ctx" not in tool.metadata.get_parameters_dict()["properties"]
