import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------------
# 1) DATA LOADING / TOKENIZER  (same as your original code, lightly cleaned)
# --------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)} # type: ignore
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
#print(data[:100])
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]           # renamed from `test_data` -> `val_data` (standard naming)

# --------------------------------------------------------------------------
# 2) HYPERPARAMETERS
#    These control model *capacity*. Small numbers here = fast to train on
#    CPU but weak model. Karpathy's video scales these up once GPU is used.
# --------------------------------------------------------------------------
batch_size = 32     # how many independent sequences we process in parallel
block_size = 128     # maximum context length (how far back the model can look)
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128         # size of each token's embedding vector (the "residual stream" width)
n_head = 4          # number of attention heads
n_layer = 4          # number of transformer blocks stacked
dropout = 0.2

torch.manual_seed(1337)

# --------------------------------------------------------------------------
# 3) BATCHING (same idea as before, now returns tensors on `device`)
# --------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    print(x[:1])
    print(y[:1])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()  # tells autograd not to track gradients here -> faster, less memory
def estimate_loss(model):
    """Averages loss over `eval_iters` batches for both splits, so the
    number we print isn't just noisy single-batch loss."""
    out = {}
    model.eval()  # switches off dropout for evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # switch dropout back on for training
    return out

# --------------------------------------------------------------------------
# 4) ONE HEAD OF SELF-ATTENTION
# --------------------------------------------------------------------------
class Head(nn.Module):
    """A single self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        # These are just linear layers -- no bias, purely a projection matrix.
        # Every token's n_embd-dim vector gets projected down to head_size.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril isn't a learnable parameter, so we register it as a "buffer"
        # (moves with .to(device), saved in state_dict, but never updated by the optimizer)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -- Batch, Time (sequence position), Channels (n_embd)
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size) -- "what do I contain"
        q = self.query(x)   # (B, T, head_size) -- "what am I looking for"

        # compute attention scores ("affinities") between every pair of tokens
        # (B,T,hs) @ (B,hs,T) -> (B,T,T): wei[b,i,j] = how much token i attends to token j
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # ^ the C**-0.5 scaling (1/sqrt(head_size)) keeps the dot products from
        # growing too large in magnitude as head_size grows, which would push
        # softmax into near one-hot, saturated regions with tiny gradients.
        # This is exactly the "Scaled" in "Scaled Dot-Product Attention".

        # causal mask: token i can only attend to tokens j <= i (no peeking at the future)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore
        wei = F.softmax(wei, dim=-1)  # normalize each row into a probability distribution
        wei = self.dropout(wei)       # randomly zero some attention weights (regularization)

        v = self.value(x)   # (B, T, head_size) -- "what I actually communicate"
        out = wei @ v        # (B,T,T) @ (B,T,hs) -> (B,T,hs): weighted sum of values
        return out

# --------------------------------------------------------------------------
# 5) MULTI-HEAD ATTENTION
#    Instead of one attention computation, run several in parallel, each
#    with its own Q/K/V projections -> each head can learn to track a
#    different kind of relationship (e.g. one head: "previous vowel",
#    another head: "start of word", etc). Concatenate the results.
# --------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # mixes information across heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run every head, each returns (B,T,head_size); concatenate along channel dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd) since num_heads*head_size = n_embd
        out = self.dropout(self.proj(out))
        return out

# --------------------------------------------------------------------------
# 6) FEEDFORWARD
#    After tokens have gathered information from each other via attention,
#    this lets each token "think" about what it gathered, independently.
#    Attention = communication between tokens. FeedForward = computation per token.
# --------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand (the "4x" is the value used in the original Transformer paper)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------
# 7) TRANSFORMER BLOCK
#    One block = communication (multi-head attention) + computation (feedforward),
#    each wrapped with a residual ("skip") connection and pre-LayerNorm.
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "x +" is the residual connection: it lets gradients flow directly
        # through the network unimpeded, which is what makes deep stacks of
        # these blocks trainable at all (without it, deep nets are very hard to optimize).
        # LayerNorm is applied BEFORE the sub-layer (this is called "pre-norm"),
        # which normalizes each token's vector to stabilize training.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --------------------------------------------------------------------------
# 8) THE FULL MODEL
# --------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off a learned vector from this table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each POSITION (0..block_size-1) also gets a learned vector, so the
        # model knows where in the sequence a token sits (attention itself
        # has no notion of order -- it's permutation-invariant without this)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layernorm before prediction
        self.lm_head = nn.Linear(n_embd, vocab_size)  # projects back up to vocab-sized logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                      # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,n_embd)
        x = tok_emb + pos_emb   # broadcasting adds position info to every batch row -> (B,T,n_embd)
        x = self.blocks(x)      # pass through all transformer blocks -> (B,T,n_embd)
        x = self.ln_f(x)        # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens -- because position
            # embedding table only has entries for 0..block_size-1, and
            # attention's tril buffer is also only block_size x block_size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]              # only need the prediction for the *next* token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --------------------------------------------------------------------------
# 9) TRAINING LOOP
# --------------------------------------------------------------------------
model = GPTLanguageModel(vocab_size)
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------------------------------------------------------
# 10) GENERATE FROM THE TRAINED MODEL
# --------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(context)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))