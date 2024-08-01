import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from functools import partial

import equinox as eqx
import jax
from absl.testing import absltest, parameterized
from jaxtyping import Array, PRNGKeyArray, PyTree

from scalax.sharding import MeshShardingHelper

# ruff: noqa: E731


class LanguageModel(eqx.Module):
    shared: eqx.nn.Shared
    attn: eqx.Module

    def __init__(self, key: PRNGKeyArray, in_dim: int, out_dim: int, n_heads: int = 4):
        embedding = eqx.nn.Embedding(out_dim, in_dim, key=key)
        linear = eqx.nn.Linear(in_dim, out_dim, key=key)

        # These two weights will now be tied together.
        where = lambda embed_and_lin: embed_and_lin[1].weight
        get = lambda embed_and_lin: embed_and_lin[0].weight

        self.shared = eqx.nn.Shared((embedding, linear), where, get)
        self.attn = eqx.nn.MultiheadAttention(n_heads, out_dim, dropout_p=0.1, key=key)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        # Expand back out so we can evaluate these layers.
        embedding, linear = self.shared()

        x = linear(embedding(x))

        return self.attn(x, x, x, key=key)


class EquinoxHelperTest(parameterized.TestCase):
    @parameterized.parameters(
        (32, 128, 4),
        (64, 256, 8),
    )
    def test_filter_sjit(self, in_dim: int, out_dim: int, n_heads: int):
        key = jax.random.PRNGKey(0)
        model = LanguageModel(key, in_dim, out_dim, n_heads)
        mesh = MeshShardingHelper(axis_dims=[-1], axis_names=["data"])

        @mesh.filter_sjit
        def identity(tree: PyTree) -> PyTree:
            return jax.tree_util.tree_map(lambda x: x, tree)

        def correct_identity(tree: PyTree) -> PyTree:
            array_tree, static_tree = eqx.partition(tree, eqx.is_array)

            @partial(mesh.sjit, static_argnums=(1,))
            def wrapped(array_tree, static_tree):
                tree = eqx.combine(array_tree, static_tree)
                mapped = jax.tree_util.tree_map(lambda x: x, tree)
                array_tree, static_tree = eqx.partition(mapped, eqx.is_array)
                return array_tree

            array_tree = wrapped(array_tree, static_tree)
            return eqx.combine(array_tree, static_tree)

        filter_out = identity(model)
        correct_out = correct_identity(model)

        self.assertIs(bool(eqx.tree_equal(filter_out[0], correct_out)), True)


if __name__ == "__main__":
    absltest.main()
