import flax
import functools

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)