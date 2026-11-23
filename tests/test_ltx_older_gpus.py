import torch

from kukkiri.ltx import _fp16_attention, _fp32_norms


class PixelNorm(torch.nn.Module):
    def forward(self, x):
        return x / torch.sqrt((x * x).mean(1, keepdim=True) + 1e-8)


def test_norms_in_fp32_do_not_overflow_fp16():
    x = torch.full((1, 4, 3), 300.0, dtype=torch.float16)
    holder = torch.nn.Sequential(PixelNorm())
    assert holder(x).float().abs().max() < 0.5
    _fp32_norms(holder)
    y = holder(x)
    assert y.dtype == torch.float16 and torch.allclose(y.float(), torch.ones_like(y.float()), atol=1e-2)


def test_fp16_attention_matches_and_keeps_the_dtype():
    def attention(q, k, v, heads, mask=None):
        b, t, hd = q.shape
        dt = q.dtype
        q, k, v = (x.view(b, -1, heads, hd // heads).transpose(1, 2).float() for x in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, t, hd).to(dt)

    holder = torch.nn.Module()
    holder.attention_function = attention
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 50, 16, dtype=torch.bfloat16) for _ in range(3))
    want = attention(q, k, v, 4)
    _fp16_attention(holder)
    got = holder.attention_function(q, k, v, 4)
    assert got.dtype == torch.bfloat16 and (got.float() - want.float()).abs().max() < 0.05
