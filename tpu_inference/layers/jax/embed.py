# Copyright 2026 Google LLC
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

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


class JaxEmbed(nnx.Embed, JaxModule):
    """Embedding layer for JAX."""

    def __init__(self,
                 *args,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        # nnx.Embed uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility, but forward it
        # as `param_dtype` so that weights are created with the correct dtype.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.Embed.__init__(self, *args, **kwargs)
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.embedding` to
        # `self.weight` such that `named_parameters()` can match the names in HF models.
        self.weight = self.embedding
        delattr(self, 'embedding')

        self.quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
            if quant_method is not None:
                self.quant_method = quant_method
                assert isinstance(quant_method, QuantizeMethodBase)
                quant_method.create_weights_jax(self)

    def __getattr__(self, name: str):
        if name == "embedding":
            # nnx.Embed needs to access self.embedding
            return self.weight

    def __call__(self, x) -> jax.Array:
        if getattr(self, "int8_group", 0):
            return self._int8_rows(x)
        if self.quant_method is None:
            return super().__call__(x)
        return self.quant_method.apply_jax(self, x)

    def decode(self, x: jax.Array) -> jax.Array:
        if getattr(self, "int8_group", 0):
            return self._int8_decode(x)
        if self.quant_method is not None:
            return self.quant_method.decode(self, x)
        return jax.numpy.dot(x, self.weight.value.T)

    # int8 storage (EMBED_INT8_GROUP): values [V, D] int8 and one bf16 scale
    # per `int8_group` columns [V, D // group], symmetric, scale = amax / 127.
    _DECODE_CHUNK = 8192

    def quantize_int8(self, group: int) -> None:
        """Replaces the loaded table with int8 values and group scales.

        Quantizes on the host in row chunks and puts only the int8 table back,
        so the device never holds both copies.
        """
        w = self.weight.value
        v, d = w.shape
        if d % group:
            raise ValueError(f"embedding width {d} is not a multiple of {group}")
        dtype, sharding = w.dtype, w.sharding
        host = np.asarray(jax.device_get(w))
        w.delete()
        q = np.empty((v, d), np.int8)
        s = np.empty((v, d // group), jnp.bfloat16)
        for r in range(0, v, 16384):
            blk = host[r:r + 16384].astype(np.float32).reshape(-1, d // group,
                                                               group)
            amax = np.abs(blk).max(axis=-1)
            sc = np.where(amax > 0, amax / 127.0, 1.0).astype(jnp.bfloat16)
            s[r:r + 16384] = sc
            q[r:r + 16384] = np.clip(
                np.rint(blk / sc.astype(np.float32)[..., None]), -127,
                127).astype(np.int8).reshape(-1, d)
        del host
        self.weight = nnx.Param(jnp.zeros((1, d), dtype))
        self.weight_q = nnx.Param(jax.device_put(q, sharding))
        self.weight_scale = nnx.Param(jax.device_put(s, sharding))
        self.int8_group = group
        self.int8_dtype = dtype

    def _dequant(self, q: jax.Array, s: jax.Array) -> jax.Array:
        g = self.int8_group
        lead, d = q.shape[:-1], q.shape[-1]
        w = (q.reshape(*lead, d // g, g).astype(jnp.float32) *
             s.astype(jnp.float32)[..., None])
        return w.reshape(*lead, d).astype(self.int8_dtype)

    def _int8_rows(self, ids: jax.Array) -> jax.Array:
        return self._dequant(jnp.take(self.weight_q.value, ids, axis=0),
                             jnp.take(self.weight_scale.value, ids, axis=0))

    def _int8_decode(self, x: jax.Array) -> jax.Array:
        """x @ W.T with W dequantized a vocabulary chunk at a time."""
        q, s = self.weight_q.value, self.weight_scale.value
        v, d = q.shape
        c = self._DECODE_CHUNK if v % self._DECODE_CHUNK == 0 else v
        n = v // c

        def chunk(args):
            qc, sc = args
            return jnp.dot(x, self._dequant(qc, sc).T)

        out = jax.lax.map(chunk, (q.reshape(n, c, d), s.reshape(n, c, -1)))
        return jnp.moveaxis(out, 0, -2).reshape(*x.shape[:-1], v)
