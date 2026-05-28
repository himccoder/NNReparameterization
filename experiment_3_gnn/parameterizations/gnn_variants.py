# experiment_3_gnn/parameterizations/gnn_variants.py
# ─────────────────────────────────────────────────────────────────────────────
# Graph Neural Network parameterizations for Experiment 3.
#
# Three GNN architectures, all implemented from scratch using only PyTorch
# (no PyTorch Geometric required). The structural optimization grid is treated
# as a 4-connected graph: each finite element is a node, and shared boundaries
# are edges. This maps naturally to FEM, where element adjacency determines
# how stresses and forces propagate.
#
# G1 — Flat GCN (FlatGCN):
#   Standard GCN with symmetric aggregation. 6 message-passing layers, 64
#   hidden channels, no spatial hierarchy. Each node aggregates from its
#   4 neighbours and itself with equal weights (mean pooling), then applies
#   a shared linear transform + BatchNorm + ReLU.
#
#   Tests: locality + weight sharing alone, without multi-scale structure.
#
# G2 — Directional GCN (DirectionalFlatGCN):
#   Same depth/width as G1 but uses 5 separate weight matrices — one per
#   spatial direction (self, up, down, left, right). This gives the GNN
#   directional sensitivity analogous to a CNN's asymmetric conv kernel.
#   Boundary nodes receive zero contribution from missing directions.
#
#   Tests: whether adding direction awareness to flat GCN closes the gap.
#
# G3 — Hierarchical GNN (HierarchicalGNN):
#   3-level encoder-decoder (Graph U-Net style): GCN layers alternate with
#   spatial average-pooling (down) and bilinear interpolation (up). Skip
#   connections concatenate encoder features into the decoder — exactly
#   mirroring the CNN U-Net. Channel progression: 1→32→64→128→256→128→64→32→1.
#
#   Tests: whether replicating the full U-Net structure via GCN layers
#   matches the CNN's performance. G3 vs R0 isolates the effect of
#   convolution (spatial filter patterns) vs graph aggregation (mean pooling).
#
# Parameter counts (approximate):
#   R0 (CNN reference): ~949k
#   G1 (Flat GCN):      ~21k   (45× fewer — smaller because no 3×3 kernels)
#   G2 (Dir. GCN):      ~103k  (9× fewer — 5 weight matrices per layer)
#   G3 (Hier. GNN):     ~109k  (9× fewer — same channel structure as CNN)
#
# The parameter count difference reflects a fundamental property: CNN conv
# layers encode 3×3=9 spatial filter patterns per (in_ch, out_ch) pair,
# while GCN linear layers encode only 1 aggregation transform. This is the
# inductive bias under test: if G3 (109k params) matches R0 (949k params),
# it suggests the hierarchical graph structure is sufficient and the specific
# convolutional weighting patterns are redundant.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GRID GRAPH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _build_gcn_adjacency(H, W):
    """
    Build edge list for a 4-connected H×W grid with self-loops.

    Nodes are indexed row-major: node at (r, c) has index r*W + c.
    Edges are undirected: for each pair of adjacent nodes (i, j), both
    (i→j) and (j→i) appear. Self-loops (i→i) are also included.

    Used by FlatGCN (G1) and HierarchicalGNN (G3) at each resolution level.

    Returns:
        src_idx: (E,) LongTensor — source node for each edge
        dst_idx: (E,) LongTensor — destination node for each edge
        degree:  (N,) FloatTensor — count of incoming edges per node
    """
    node_idx = torch.arange(H * W).reshape(H, W)
    srcs, dsts = [], []

    # Self-loops: every node receives from itself
    flat = node_idx.reshape(-1)
    srcs.append(flat)
    dsts.append(flat)

    # Horizontal pairs (r, c) ↔ (r, c+1)
    s = node_idx[:, :-1].reshape(-1)   # left node
    d = node_idx[:, 1:].reshape(-1)    # right node
    srcs.extend([s, d])
    dsts.extend([d, s])

    # Vertical pairs (r, c) ↔ (r+1, c)
    s = node_idx[:-1, :].reshape(-1)   # top node
    d = node_idx[1:, :].reshape(-1)    # bottom node
    srcs.extend([s, d])
    dsts.extend([d, s])

    src_idx = torch.cat(srcs)
    dst_idx = torch.cat(dsts)

    degree = torch.zeros(H * W, dtype=torch.float32)
    degree.scatter_add_(0, dst_idx, torch.ones(len(dst_idx), dtype=torch.float32))

    return src_idx, dst_idx, degree


def _build_directional_adjacency(H, W):
    """
    Build per-direction neighbor lookup for a 4-connected H×W grid.

    For each node i, returns the index of its neighbour in each cardinal
    direction. Boundary nodes (e.g. top row has no up-neighbour) use
    index N = H*W as a sentinel. The GNN pads the feature matrix with a
    zero row at index N so boundary nodes receive zero contribution.

    Returns:
        up_idx, down_idx, left_idx, right_idx: each a (N,) LongTensor.
            Value N means "no neighbour in this direction".
    """
    N = H * W
    node_idx = torch.arange(N).reshape(H, W)

    # up_idx[i] = index of node above i; N if i is in the top row
    up_idx = torch.full((N,), N, dtype=torch.long)
    # nodes in rows 1..H-1 have an up-neighbour in rows 0..H-2
    dst_rows = node_idx[1:, :].reshape(-1)   # nodes that have an upper neighbour
    src_rows = node_idx[:-1, :].reshape(-1)  # their upper neighbours
    up_idx[dst_rows] = src_rows

    down_idx = torch.full((N,), N, dtype=torch.long)
    dst_rows = node_idx[:-1, :].reshape(-1)
    src_rows = node_idx[1:, :].reshape(-1)
    down_idx[dst_rows] = src_rows

    left_idx = torch.full((N,), N, dtype=torch.long)
    dst_cols = node_idx[:, 1:].reshape(-1)   # nodes that have a left neighbour
    src_cols = node_idx[:, :-1].reshape(-1)  # their left neighbours
    left_idx[dst_cols] = src_cols

    right_idx = torch.full((N,), N, dtype=torch.long)
    dst_cols = node_idx[:, :-1].reshape(-1)
    src_cols = node_idx[:, 1:].reshape(-1)
    right_idx[dst_cols] = src_cols

    return up_idx, down_idx, left_idx, right_idx


def _pool_grid(h, H, W):
    """
    Average-pool node features from (H*W, F) to ((H//2)*(W//2), F).

    Reshapes node features into a spatial (H, W) grid and applies
    2×2 average pooling, mirroring MaxPool2d in the CNN encoder.
    The output grid has size floor(H/2) × floor(W/2).
    """
    _, Fc = h.shape
    # (H*W, Fc) → (H, W, Fc) → (1, Fc, H, W) for avg_pool2d
    x = h.reshape(H, W, Fc).permute(2, 0, 1).unsqueeze(0)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    H2, W2 = x.shape[2], x.shape[3]
    return x.squeeze(0).permute(1, 2, 0).reshape(H2 * W2, Fc), H2, W2


def _unpool_grid(h, H_src, W_src, H_dst, W_dst):
    """
    Upsample node features from (H_src*W_src, F) to (H_dst*W_dst, F).

    Uses bilinear interpolation (same as CNN's _match_and_concat), so
    the output exactly matches the target resolution even for odd dimensions.
    """
    _, Fc = h.shape
    x = h.reshape(H_src, W_src, Fc).permute(2, 0, 1).unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x, size=(H_dst, W_dst), mode="bilinear", align_corners=False
    )
    return x.squeeze(0).permute(1, 2, 0).reshape(H_dst * W_dst, Fc)


# ─────────────────────────────────────────────────────────────────────────────
# GCN LAYER — Standard symmetric aggregation
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single GCN message-passing layer.

    Aggregation: for each node i, compute the mean of features from all
    incoming edges (neighbours + self). Then apply a shared linear
    transform, optional BatchNorm, and ReLU.

    h_i^(l+1) = ReLU( BN( W · mean_{j ∈ N(i) ∪ {i}} h_j^(l) ) )

    The weight matrix W is shared across all nodes (translation equivariance),
    analogous to CNN weight sharing.
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        # No bias when BatchNorm follows (BN has its own bias)
        self.linear = nn.Linear(in_channels, out_channels, bias=not use_batchnorm)
        self.bn     = nn.BatchNorm1d(out_channels) if use_batchnorm else None
        self.act    = nn.ReLU(inplace=True)

    def forward(self, h, src_idx, dst_idx, degree):
        """
        Args:
            h:       (N, F_in) node features
            src_idx: (E,) source node for each edge (precomputed from grid)
            dst_idx: (E,) destination node for each edge
            degree:  (N,) number of incoming edges per node (for normalisation)

        Returns:
            (N, F_out) updated node features
        """
        N, F = h.shape
        # Gather source features for all edges
        h_src = h[src_idx]                                      # (E, F)
        # Scatter-add: sum messages arriving at each destination node
        idx_exp = dst_idx.unsqueeze(1).expand(-1, F)            # (E, F)
        agg = torch.zeros(N, F, device=h.device, dtype=h.dtype)
        agg = agg.scatter_add(0, idx_exp, h_src)                # (N, F)
        # Normalise by degree (mean aggregation)
        agg = agg / degree.to(h.device).unsqueeze(1)            # (N, F)

        out = self.linear(agg)
        if self.bn is not None:
            out = self.bn(out)
        return self.act(out)


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTIONAL GCN LAYER — Separate weight per spatial direction
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalGCNLayer(nn.Module):
    """
    Direction-aware GCN layer with 5 separate weight matrices.

    Each of the 5 spatial directions (self, up, down, left, right) has its
    own linear transform. Boundary nodes receive zero contribution from
    missing directions. This is the closest GNN analogue to a CNN conv
    layer: the spatial weight pattern is explicit (5 directions vs 3×3 kernel).

    h_i^(l+1) = ReLU( BN(
        W_self · h_i
      + W_up   · h_{up(i)}   [or 0 if top row]
      + W_down · h_{down(i)} [or 0 if bottom row]
      + W_left · h_{left(i)} [or 0 if left column]
      + W_right· h_{right(i)}[or 0 if right column]
    ))
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        bias = not use_batchnorm
        self.W_self  = nn.Linear(in_channels, out_channels, bias=bias)
        self.W_up    = nn.Linear(in_channels, out_channels, bias=False)
        self.W_down  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_left  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_right = nn.Linear(in_channels, out_channels, bias=False)
        self.bn  = nn.BatchNorm1d(out_channels) if use_batchnorm else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, h, up_idx, down_idx, left_idx, right_idx):
        """
        Args:
            h:         (N, F_in) node features
            up_idx:    (N,) index of up-neighbour; value N = "no neighbour"
            down_idx:  (N,) index of down-neighbour; value N = "no neighbour"
            left_idx:  (N,) index of left-neighbour; value N = "no neighbour"
            right_idx: (N,) index of right-neighbour; value N = "no neighbour"

        Returns:
            (N, F_out) updated node features
        """
        # Pad with a zero row so sentinel index N safely returns zeros
        zeros_row = torch.zeros(1, h.shape[1], device=h.device, dtype=h.dtype)
        h_pad = torch.cat([h, zeros_row], dim=0)    # (N+1, F_in)

        h_up    = h_pad[up_idx.to(h.device)]        # (N, F_in)
        h_down  = h_pad[down_idx.to(h.device)]
        h_left  = h_pad[left_idx.to(h.device)]
        h_right = h_pad[right_idx.to(h.device)]

        out = (
            self.W_self(h)
            + self.W_up(h_up)
            + self.W_down(h_down)
            + self.W_left(h_left)
            + self.W_right(h_right)
        )                                            # (N, F_out)

        if self.bn is not None:
            out = self.bn(out)
        return self.act(out)


# ─────────────────────────────────────────────────────────────────────────────
# G1 — FLAT GCN  (no spatial hierarchy)
# ─────────────────────────────────────────────────────────────────────────────

class FlatGCN(nn.Module):
    """
    G1: 6-layer GCN on the full-resolution 4-connected grid.

    Architecture:
        z (N, 1)
        → GCNLayer(1, hidden)            # input projection + neighbourhood agg
        → GCNLayer(hidden, hidden) × 5   # deep message-passing
        → Linear(hidden, 1) + Sigmoid    # density output
        → reshape (H, W)

    No spatial pooling: every layer operates at full resolution.
    Receptive field after 6 hops ≈ 6 elements in each direction (~12×12 area),
    which covers most of the MBB beam (25 tall) and the full causeway bridge
    width after ~48 hops (i.e. information diffuses fully after ~48 steps of
    the GCN iteration, but 6 layers limits it).
    """

    def __init__(self, H, W, hidden_channels=64, n_layers=6, use_batchnorm=True):
        super().__init__()
        self.H = H
        self.W = W
        self.hidden = hidden_channels

        # Build and register grid adjacency as buffers (not optimized)
        src_idx, dst_idx, degree = _build_gcn_adjacency(H, W)
        self.register_buffer("src_idx", src_idx)
        self.register_buffer("dst_idx", dst_idx)
        self.register_buffer("degree",  degree)

        # Input layer: project 1-dim latent to hidden channels via GCN
        self.input_layer = GCNLayer(1, hidden_channels, use_batchnorm=use_batchnorm)

        # Hidden layers
        self.layers = nn.ModuleList([
            GCNLayer(hidden_channels, hidden_channels, use_batchnorm=use_batchnorm)
            for _ in range(n_layers - 1)
        ])

        # Output: linear + sigmoid (no BN on final layer)
        self.output_proj = nn.Linear(hidden_channels, 1)

    def forward(self, z):
        """
        Args:
            z: (N, 1) per-node latent features

        Returns:
            density: (H, W) float tensor in [0, 1]
        """
        h = self.input_layer(z, self.src_idx, self.dst_idx, self.degree)
        for layer in self.layers:
            h = layer(h, self.src_idx, self.dst_idx, self.degree)
        out = torch.sigmoid(self.output_proj(h))   # (N, 1)
        return out.reshape(self.H, self.W)


# ─────────────────────────────────────────────────────────────────────────────
# G2 — DIRECTIONAL FLAT GCN  (direction-sensitive, no hierarchy)
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalFlatGCN(nn.Module):
    """
    G2: 6-layer directional GCN with per-direction weight matrices.

    Same depth and hidden dimension as G1 (FlatGCN), but each layer uses
    5 separate weight matrices instead of 1. This gives the network explicit
    awareness of which neighbour is to the left, right, above, or below —
    the same directional information that CNN conv filters encode via their
    spatial kernel arrangement (e.g. a 3×3 kernel with 9 independently-
    weighted positions encodes direction; the directional GCN approximates
    this with 5 positions).

    Architecture:
        z (N, 1)
        → DirectionalGCNLayer(1, hidden) × 1
        → DirectionalGCNLayer(hidden, hidden) × 5
        → Linear(hidden, 1) + Sigmoid
        → reshape (H, W)

    ~5× more parameters per layer than G1 (5 weight matrices vs 1).
    """

    def __init__(self, H, W, hidden_channels=64, n_layers=6, use_batchnorm=True):
        super().__init__()
        self.H = H
        self.W = W

        # Build and register directional adjacency as buffers
        up_idx, down_idx, left_idx, right_idx = _build_directional_adjacency(H, W)
        self.register_buffer("up_idx",    up_idx)
        self.register_buffer("down_idx",  down_idx)
        self.register_buffer("left_idx",  left_idx)
        self.register_buffer("right_idx", right_idx)

        self.input_layer = DirectionalGCNLayer(
            1, hidden_channels, use_batchnorm=use_batchnorm
        )
        self.layers = nn.ModuleList([
            DirectionalGCNLayer(
                hidden_channels, hidden_channels, use_batchnorm=use_batchnorm
            )
            for _ in range(n_layers - 1)
        ])
        self.output_proj = nn.Linear(hidden_channels, 1)

    def forward(self, z):
        """
        Args:
            z: (N, 1) per-node latent features

        Returns:
            density: (H, W) float tensor in [0, 1]
        """
        dirs = (self.up_idx, self.down_idx, self.left_idx, self.right_idx)
        h = self.input_layer(z, *dirs)
        for layer in self.layers:
            h = layer(h, *dirs)
        out = torch.sigmoid(self.output_proj(h))
        return out.reshape(self.H, self.W)


# ─────────────────────────────────────────────────────────────────────────────
# G3 — HIERARCHICAL GNN  (Graph U-Net)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalGNN(nn.Module):
    """
    G3: 3-level hierarchical GNN mirroring the CNN U-Net structure.

    Encoder: alternating GCN layers and spatial average-pooling (2×2).
    Decoder: alternating GCN layers and bilinear upsampling with skip
             connections (concatenation of encoder features).

    Channel progression (c = base_channels = 32):
        Enc1: (N0, 1)  → GCN → (N0, c)         [full res   H0×W0]
        Pool: (N0, c)  → pool → (N1, c)
        Enc2: (N1, c)  → GCN → (N1, 2c)        [half res   H1×W1]
        Pool: (N1, 2c) → pool → (N2, 2c)
        Enc3: (N2, 2c) → GCN → (N2, 4c)        [quarter    H2×W2]
        Pool: (N2, 4c) → pool → (N3, 4c)
        BN:   (N3, 4c) → GCN → (N3, 8c)        [eighth     H3×W3]

        Unpool to H2×W2: (N2, 8c)
        Cat skip enc3: (N2, 8c+4c) → Dec3 GCN → (N2, 4c)
        Unpool to H1×W1: (N1, 4c)
        Cat skip enc2: (N1, 4c+2c) → Dec2 GCN → (N1, 2c)
        Unpool to H0×W0: (N0, 2c)
        Cat skip enc1: (N0, 2c+c)  → Dec1 GCN → (N0, c)
        Linear(c, 1) + Sigmoid → (H0, W0)

    This is the direct graph analogue of the CNN StandardUNet:
      - ConvBlock(A, B) replaced by GCNLayer(A, B)
      - MaxPool2d replaced by avg_pool2d (spatial, over grid nodes)
      - ConvTranspose2d replaced by bilinear interpolation
      - Skip connections and channel progression are identical
    """

    def __init__(self, H, W, base_channels=32, use_batchnorm=True):
        super().__init__()
        c  = base_channels
        bn = use_batchnorm

        # Store full-resolution dimensions
        self.H0, self.W0 = H, W

        # Compute level dimensions (floor division, same as MaxPool2d)
        self.H1, self.W1 = H  // 2, W  // 2
        self.H2, self.W2 = H  // 4, W  // 4
        self.H3, self.W3 = H  // 8, W  // 8

        # Build and register GCN adjacency at each resolution level
        for level, (Hl, Wl) in enumerate([
            (self.H0, self.W0),
            (self.H1, self.W1),
            (self.H2, self.W2),
            (self.H3, self.W3),
        ]):
            src, dst, deg = _build_gcn_adjacency(Hl, Wl)
            self.register_buffer(f"src_{level}", src)
            self.register_buffer(f"dst_{level}", dst)
            self.register_buffer(f"deg_{level}", deg)

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = GCNLayer(1,     c,    use_batchnorm=bn)   # at L0
        self.enc2 = GCNLayer(c,     c*2,  use_batchnorm=bn)   # at L1
        self.enc3 = GCNLayer(c*2,   c*4,  use_batchnorm=bn)   # at L2

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = GCNLayer(c*4, c*8, use_batchnorm=bn)  # at L3

        # ── Decoder ──────────────────────────────────────────────────────
        # After unpool + skip concat, input channels are doubled
        self.dec3 = GCNLayer(c*8 + c*4, c*4, use_batchnorm=bn)  # at L2
        self.dec2 = GCNLayer(c*4 + c*2, c*2, use_batchnorm=bn)  # at L1
        self.dec1 = GCNLayer(c*2 + c,   c,   use_batchnorm=bn)  # at L0

        # ── Output ───────────────────────────────────────────────────────
        self.final = nn.Linear(c, 1)

    def _adj(self, level):
        """Return (src_idx, dst_idx, degree) buffers for a given level."""
        return (
            getattr(self, f"src_{level}"),
            getattr(self, f"dst_{level}"),
            getattr(self, f"deg_{level}"),
        )

    def forward(self, z):
        """
        Args:
            z: (N0, 1) per-node latent features at full resolution

        Returns:
            density: (H0, W0) float tensor in [0, 1]
        """
        # ── Encoder ──────────────────────────────────────────────────────
        e1 = self.enc1(z, *self._adj(0))                            # (N0, c)
        z1, H1, W1 = _pool_grid(e1, self.H0, self.W0)              # (N1, c)

        e2 = self.enc2(z1, *self._adj(1))                           # (N1, 2c)
        z2, H2, W2 = _pool_grid(e2, H1, W1)                        # (N2, 2c)

        e3 = self.enc3(z2, *self._adj(2))                           # (N2, 4c)
        z3, H3, W3 = _pool_grid(e3, H2, W2)                        # (N3, 4c)

        # ── Bottleneck ───────────────────────────────────────────────────
        b = self.bottleneck(z3, *self._adj(3))                      # (N3, 8c)

        # ── Decoder ──────────────────────────────────────────────────────
        b_up  = _unpool_grid(b,    H3, W3, H2, W2)                 # (N2, 8c)
        d3    = self.dec3(torch.cat([b_up, e3], dim=1), *self._adj(2))  # (N2, 4c)

        d3_up = _unpool_grid(d3,   H2, W2, H1, W1)                 # (N1, 4c)
        d2    = self.dec2(torch.cat([d3_up, e2], dim=1), *self._adj(1)) # (N1, 2c)

        d2_up = _unpool_grid(d2,   H1, W1, self.H0, self.W0)       # (N0, 2c)
        d1    = self.dec1(torch.cat([d2_up, e1], dim=1), *self._adj(0)) # (N0, c)

        # ── Output ───────────────────────────────────────────────────────
        out = torch.sigmoid(self.final(d1))                         # (N0, 1)
        return out.reshape(self.H0, self.W0)                        # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_gnn(gnn_type, H, W, **kwargs):
    """
    Instantiate a GNN architecture by name.

    Args:
        gnn_type: "flat_gcn" | "directional_gcn" | "hierarchical_gnn"
        H, W:     grid height and width
        **kwargs: architecture hyperparameters (hidden, n_layers, base_channels, use_batchnorm)

    Returns:
        nn.Module
    """
    if gnn_type == "flat_gcn":
        return FlatGCN(
            H             = H,
            W             = W,
            hidden_channels = kwargs.get("hidden", 64),
            n_layers      = kwargs.get("n_layers", 6),
            use_batchnorm = kwargs.get("use_batchnorm", True),
        )
    elif gnn_type == "directional_gcn":
        return DirectionalFlatGCN(
            H             = H,
            W             = W,
            hidden_channels = kwargs.get("hidden", 64),
            n_layers      = kwargs.get("n_layers", 6),
            use_batchnorm = kwargs.get("use_batchnorm", True),
        )
    elif gnn_type == "hierarchical_gnn":
        return HierarchicalGNN(
            H             = H,
            W             = W,
            base_channels = kwargs.get("base_channels", 32),
            use_batchnorm = kwargs.get("use_batchnorm", True),
        )
    else:
        raise ValueError(
            f"Unknown gnn_type '{gnn_type}'. "
            f"Choose from: flat_gcn, directional_gcn, hierarchical_gnn."
        )


# ─────────────────────────────────────────────────────────────────────────────
# GNN PARAMETERIZATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class GNNParameterization:
    """
    Wraps GNN architectures for use by the Experiment 3 runner.

    Convention (analogous to CNN in Experiments 1 & 2):
      - A fixed random latent z of shape (N, 1) = (H*W, 1) is the GNN input,
        initialized from seed 42 (same seed as CNN latent for comparability).
      - GNN weights W are the optimization variables.
      - z stays fixed throughout optimization.

    Interface compatible with run_gradient_optimizer() in optimizers/.
    """

    def __init__(self, args, gnn_type, **kwargs):
        """
        Args:
            args:     problem ObjectView (contains nelx, nely, density)
            gnn_type: "flat_gcn" | "directional_gcn" | "hierarchical_gnn"
            **kwargs: passed through to build_gnn()
        """
        self.args     = args
        self.nely     = args.nely
        self.nelx     = args.nelx
        self.gnn_type = gnn_type
        self.kwargs   = kwargs

        self.model = build_gnn(gnn_type, args.nely, args.nelx, **kwargs)

        # Fixed random latent — same seed as CNN experiments for comparability.
        # Shape (N, 1) where N = nely * nelx.  The values are the same as the
        # CNN's z = randn(1, 1, nely, nelx) since both draw H*W samples from
        # RandomState(42).
        N  = args.nely * args.nelx
        rng = np.random.RandomState(42)
        z_np = rng.randn(N, 1).astype(np.float32)
        self.z = torch.from_numpy(z_np)                 # (N, 1)

        self._n_params = sum(p.numel() for p in self.model.parameters())

    # ── Interface expected by run_gradient_optimizer ──────────────────────

    def initial_params(self):
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def to_density(self, params_vec):
        """Forward pass (no grad): flat params → (nely, nelx) density array."""
        self._load_params(params_vec)
        with torch.no_grad():
            density = self.model(self.z)    # (H, W)
        # Ensure exact output size (handles edge cases from pooling)
        if density.shape != (self.nely, self.nelx):
            density = torch.nn.functional.interpolate(
                density.unsqueeze(0).unsqueeze(0),
                size=(self.nely, self.nelx),
                mode="bilinear", align_corners=False,
            ).squeeze()
        return density.numpy()

    def to_density_with_grad(self, params_vec):
        """Forward pass retaining autograd graph for backpropagation."""
        self._load_params(params_vec)
        density = self.model(self.z)        # (H, W)
        if density.shape != (self.nely, self.nelx):
            density = torch.nn.functional.interpolate(
                density.unsqueeze(0).unsqueeze(0),
                size=(self.nely, self.nelx),
                mode="bilinear", align_corners=False,
            ).squeeze()
        return density

    def description(self):
        type_labels = {
            "flat_gcn":        "Flat GCN (G1)",
            "directional_gcn": "Directional GCN (G2)",
            "hierarchical_gnn":"Hierarchical GNN (G3)",
        }
        label = type_labels.get(self.gnn_type, self.gnn_type)
        kw_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return (
            f"GNN variant: {label}. "
            f"Config: {kw_str}. "
            f"Parameters: {self._n_params:,}."
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_flat_params(self):
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.model.parameters()
        ])

    def _load_params(self, flat_params):
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(
                    flat_params[offset:offset + n].reshape(p.shape).astype(np.float32)
                )
            )
            offset += n
