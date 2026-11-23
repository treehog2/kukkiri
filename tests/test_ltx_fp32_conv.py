import torch

from kukkiri.ltx import _fp32_conv3d


def test_sliced_fp32_conv3d_matches_a_whole_fp32_convolution():
    torch.manual_seed(0)
    seq = torch.nn.Sequential(torch.nn.Conv3d(4, 6, 3, padding=(0, 1, 1)), torch.nn.Conv3d(6, 5, 3, stride=(2, 1, 1), padding=(0, 1, 1)))
    x = torch.randn(1, 4, 37, 9, 9)
    want = seq(x)
    _fp32_conv3d(seq, frames=4)
    got = seq(x.bfloat16()).float()
    assert got.shape == want.shape
    assert (got - want).abs().max() < 0.05


def test_chunked_attention_matches_the_whole_call():
    from kukkiri.ltx import _chunk_attention

    def attention(q, k, v, heads, mask=None):
        b, t, hd = q.shape
        q, k, v = (x.view(b, -1, heads, hd // heads).transpose(1, 2) for x in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(b, t, hd)

    holder = torch.nn.Module()
    holder.attention_function, holder.masked_attention_function = attention, attention
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 50, 16), torch.randn(2, 30, 16), torch.randn(2, 30, 16)
    mask = torch.randn(2, 1, 50, 30)
    want, want_masked = attention(q, k, v, 4), attention(q, k, v, 4, mask)
    _chunk_attention(holder, rows=16)
    assert torch.allclose(holder.attention_function(q, k, v, 4), want, atol=1e-6)
    assert torch.allclose(holder.masked_attention_function(q, k, v, 4, mask), want_masked, atol=1e-6)
