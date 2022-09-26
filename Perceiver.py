import torch
import torch.nn as nn
from einops import rearrange, repeat
import copy
from torch.nn.functional import log_softmax


def make_layer_stack(original_layer, N):
    return nn.ModuleList([copy.deepcopy(original_layer) for _ in range(N)])


class LinSoftMax(nn.Module):
    def __init__(self, dim_model, vocab_len):
        super(LinSoftMax, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_len)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Attention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim=None,
                 heads=8,
                 dim_model=512,
                 dropout_rate=0.0):
        super(Attention, self).__init__()
        key_dim = key_dim if key_dim is not None else query_dim
        self.heads = heads
        self.dim_model = dim_model

        self.make_query = nn.Linear(query_dim, dim_model, bias=False)
        self.make_kv = nn.Linear(key_dim, dim_model*2, bias=False)
        self.to_out = nn.Linear(dim_model, query_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, context=None, mask=None):
        q = self.make_query(x)
        context = context if context is not None else x
        k, v = self.make_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)*((self.dim_model // self.heads)**(-0.5))
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            sim.masked_fill_(~mask+2, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)


class Norm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super(Norm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x*nn.functional.gelu(gates)


class TransformerLayer(nn.Module):
    def __init__(self, self_attn, ff):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.ff = ff

    def forward(self, x, mask=None):
        x = self.self_attn(x) + x
        x = self.ff(x) + x
        return x


class Perceiver(nn.Module):
    def __init__(self,
                 dim,
                 N,
                 dim_queries,
                 dim_latents,
                 num_latents,
                 self_attn_heads,
                 cross_attn_heads,
                 dim_cross_attn,
                 dim_self_attn,
                 dim_logits,
                 depth,
                 dropout_rate):

        super(Perceiver, self).__init__()
        self.encoder = Encoder(dim, N, dim_queries, dim_latents, num_latents, self_attn_heads, cross_attn_heads, dim_cross_attn, dim_self_attn, dropout_rate)
        self.decoder = Decoder(dim_queries, dim_latents, cross_attn_heads, dim_cross_attn, dropout_rate)
        self.to_logits = self.to_logits = LinSoftMax(dim_queries, dim_logits)
        self.depth = depth

    def forward(self, x, mask=None, queries=None):
        for i in range(self.depth):
            x, queries = self.decoder(self.encoder(x, mask=mask), queries=queries)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 dim,
                 N,
                 dim_queries,
                 dim_latents,
                 num_latents,
                 self_attn_heads,
                 cross_attn_heads,
                 dim_cross_attn,
                 dim_self_attn,
                 dropout_rate):
        super(Encoder, self).__init__()
        latents = torch.randn(num_latents, dim_latents)
        self.latents = nn.Parameter(latents)

        cross_attn = Attention(dim_queries, key_dim=dim_latents, heads=cross_attn_heads, dim_model=dim_cross_attn, dropout_rate=dropout_rate)

        self.cross_attn = Norm(dim_latents, cross_attn, context_dim=dim)
        self.cross_attn_ff = Norm(dim_latents, FeedForward(dim_latents))

        self_attn = Attention(dim_latents, heads=self_attn_heads, dim_model=dim_self_attn)
        transformer_layer = TransformerLayer(Norm(dim_latents, self_attn), Norm(dim_latents, FeedForward(dim_latents)))

        self.transformer = make_layer_stack(transformer_layer, N)

    def forward(self, data, mask=None):
        b, *_, device = *data.shape, data.device
        x = repeat(self.latents, 'n d -> b n d', b=b)

        x = self.cross_attn(x, context=data, mask=mask) + x
        #x = self.dropout(self.cross_attn_ff(x)) + x

        for layer in self.transformer:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim_queries,
                 dim_latents,
                 cross_attn_heads,
                 dim_cross_attn,
                 dropout_rate):
        super(Decoder, self).__init__()

        cross_attn = Attention(dim_queries, key_dim=dim_latents, heads=cross_attn_heads, dim_model=dim_cross_attn, dropout_rate=dropout_rate)
        self.cross_attn = Norm(dim_queries, cross_attn, context_dim=dim_latents)
        self.cross_attn_ff = Norm(dim_queries, FeedForward(dim_queries))

    def forward(self, data, queries=None):
        x = self.cross_attn(queries, context=data)
        #x = self.dropout(self.cross_attn_ff(x)) + x

        return x, queries


class PerceiverImageClassification(nn.Module):
    def __init__(self, max_seq_len, dim, num_output_tokens, **kwargs):
        super(PerceiverImageClassification, self).__init__()

        self.perceiver = Perceiver(dim=dim, dim_queries=dim, dim_logits=num_output_tokens, **kwargs)
        self.to_logits = self.to_logits = LinSoftMax(dim, num_output_tokens)

        self.max_freq = 10.
        self.num_freq_bands = 6

    @staticmethod
    def fourier_encode(x, max_freq, num_bands=4):
        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * torch.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1)
        return x

    def forward(self, data):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        enc_pos = self.fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        data = torch.cat((data, enc_pos), dim=-1)

        x = rearrange(data, 'b ... d -> b (...) d')

        out = self.perceiver(x, queries=x)
        return out, self.to_logits(out)[:, 0, :]


def make_model_img_classification(max_seq_len,
                                  dim,
                                  num_output_tokens,
                                  N,
                                  dim_latents,
                                  num_latents,
                                  self_attn_heads,
                                  cross_attn_heads,
                                  dim_cross_attn,
                                  dim_self_attn,
                                  depth,
                                  dropout_rate):
    model = PerceiverImageClassification(
        max_seq_len=max_seq_len,
        dim=dim,
        num_output_tokens=num_output_tokens,
        N=N,
        dim_latents=dim_latents,
        num_latents=num_latents,
        self_attn_heads=self_attn_heads,
        cross_attn_heads=cross_attn_heads,
        dim_cross_attn=dim_cross_attn,
        dim_self_attn=dim_self_attn,
        depth=depth,
        dropout_rate=dropout_rate
    )

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model

