from typing import Any, List

from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion.pipeline import get_transformation_hash
from llama_index.core.schema import BaseNode, TextNode, TransformComponent


class DummyTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            node.set_content(node.get_content() + "\nTESTTEST")
        return nodes


def test_cache() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    cache.put(hash, new_nodes)

    cache_hit = cache.get(hash)
    assert cache_hit is not None
    assert cache_hit[0].get_content() == new_nodes[0].get_content()

    new_hash = get_transformation_hash(new_nodes, transformation)
    assert cache.get(new_hash) is None


def test_get_transformation_hash_is_injective() -> None:
    transformation = DummyTransform()

    # Splitting the same combined text at a different node boundary must not
    # collide: ["ab", "c"] and ["a", "bc"] previously hashed identically because
    # node contents were concatenated with no separator.
    nodes_ab_c = [TextNode(id_="1", text="ab"), TextNode(id_="2", text="c")]
    nodes_a_bc = [TextNode(id_="3", text="a"), TextNode(id_="4", text="bc")]
    assert get_transformation_hash(
        nodes_ab_c, transformation
    ) != get_transformation_hash(nodes_a_bc, transformation)

    # Two distinct documents with identical content must not hash the same,
    # since the cached value carries node identity (id_, ref_doc_id) that the
    # key must also depend on.
    node_a = TextNode(id_="doc_a", text="same content")
    node_b = TextNode(id_="doc_b", text="same content")
    assert get_transformation_hash([node_a], transformation) != get_transformation_hash(
        [node_b], transformation
    )


def test_cache_clear() -> None:
    cache = IngestionCache()
    transformation = DummyTransform()

    node = TextNode(text="dummy")
    hash = get_transformation_hash([node], transformation)

    new_nodes = transformation([node])
    cache.put(hash, new_nodes)

    cache_hit = cache.get(hash)
    assert cache_hit is not None

    cache.clear()
    assert cache.get(hash) is None
