"""
tests/test_real_pcap_pipeline.py
===============================

Test de integración usando un PCAP real para verificar que el
pipeline data + representations funciona correctamente.:

PCAP -> Flows -> Representation -> Tensor -> Image

Además muestra ejemplos de representaciones.
"""

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.preprocessing import (
    PCAPPipeline,
    PacketWindowAggregator,
    TrafficChunkAggregator
)
from src.representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from src.representations.vision import (
    GAFRepresentation, GAFConfig,
    NprintRepresentation, NprintConfig,
    NprintImageRepresentation, NprintImageConfig,
)

PCAP_PATH = Path("data/pcap/Benign/Gmail.pcap")

print(f"Usando PCAP de prueba: {PCAP_PATH}")
print("=" * 50)

# -----------------------------------------------------------
# Métricas auxiliares
# -----------------------------------------------------------

def compute_sparsity(x: np.ndarray) -> float:
    """
    Proporción de valores cercanos a cero.
    """
    if x.size == 0:
        return 0.0
    return float(np.mean(np.isclose(x, 0.0)))


def compute_entropy(x: np.ndarray, n_bins: int = 256) -> float:
    """
    Entropía discreta basada en histograma (Shannon).
    """
    x = x.flatten()

    if np.all(x == x[0]):
        return 0.0

    hist, _ = np.histogram(x, bins=n_bins, density=False)

    # Normalizar a probabilidades
    p = hist.astype(np.float64)
    p = p / p.sum()

    # Evitar log(0)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def compression_ratio(original_size: int, rep: np.ndarray) -> float:
    """
    Ratio simple: tamaño original / tamaño representación.
    """
    rep_size = rep.size
    if rep_size == 0:
        return 0.0
    return float(original_size / rep_size)

# -----------------------------------------------------------
# Visualización comparativa (MUY útil para el TFG)
# -----------------------------------------------------------
def test_visualize_flow_representations_pipeline():
    # -----------------------------
    # 1. Datos
    # -----------------------------
    flow_pipeline = PCAPPipeline(max_packets=None)
    flows = flow_pipeline.process(PCAP_PATH)
    flow = flows[0]

    window_pipeline = PCAPPipeline(
        aggregator=PacketWindowAggregator,
        max_packets=None,
        window_size=1024
    )
    windows = window_pipeline.process(PCAP_PATH)
    window = windows[0]
    
    # Pipeline específico para GAF (temporal + overlap)
    chunk_pipeline = PCAPPipeline(
        aggregator=TrafficChunkAggregator,
        max_packets=None,
        chunk_duration=1.0,
        stride=0.5   # overlap 50%
    )

    chunks = chunk_pipeline.process(PCAP_PATH)
    chunk = chunks[0]
        
    print(f"[DEBUG] flow type: {type(flow)} | num packets: {len(flow.packets)}")
    
    for i, f in enumerate(flows[:5]):
        print(f"Flow {i} | packets: {len(f.packets)} | sample iat: {[p.iat for p in f.packets[:5]]}")
    
    print(f"[DEBUG] chunk packets: {len(chunk.packets)}")
    print(f"[DEBUG] sample iat chunk: {[p.iat for p in chunk.packets[:10]]}")
    
    # Para el flujo, usamos el número de paquetes multiplicado por 1 (cada valor codificado como float)
    original_size_flow = len(flow.packets)
    # Para la ventana, número de paquetes * número de features codificadas
    original_size_window = len(window.packets)

    # -----------------------------
    # 2. Representaciones
    # -----------------------------
    tokenizer = FlatTokenizer(SequentialConfig(max_length=128))
    gasf = GAFRepresentation(GAFConfig(image_size=128, field_name="iat"))
    nprint = NprintRepresentation(NprintConfig(max_packets=1024))

    nprint_img_rep = NprintImageRepresentation(
        NprintImageConfig(
            patch_size=4,
            max_packets=128,
            use_ch3_variance=True,
            pad_to_height=128
        )
    )

    tokenizer.fit(flows[:50])
    gasf.fit(flows[:50])
    nprint.fit(windows[:50])
    nprint_img_rep.fit(windows[:50])

    tokens = tokenizer.decode(tokenizer.encode(flow))
    gasf_img = gasf.encode(chunk)[0].numpy()
    nprint_img = nprint.encode(window).numpy()
    nprint_img_compact = nprint_img_rep.encode(window).numpy()

    # -----------------------------
    # 3. TOKENIZER (figura 1)
    # -----------------------------
    plt.figure(figsize=(12, 2))
    plt.title("Tokenizer representation (sequence)")
    plt.text(
        0.01, 0.5,
        " ".join(tokens[:50]),
        fontsize=10,
        family="monospace",
        va="center"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. GASF (figura 2)
    # -----------------------------
    plt.figure(figsize=(5, 5))
    plt.title("GASF representation")
    
    # Detectar rango automáticamente según config
    if gasf.cfg.rescale_to_01:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = -1.0, 1.0

    im = plt.imshow(
        gasf_img,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )

    plt.xlabel("Time steps")
    plt.ylabel("Transformed steps")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 5. NPRINT (figura 3)
    # -----------------------------
    cmap_nprint = ListedColormap([
        "#CDCDCD",  # -1
        "#FF0000",  # 0
        "#00FF00",  # 1
    ])

    plt.figure(figsize=(10, 6))
    plt.title("Nprint representation (packets x bits)")

    plt.imshow(
        nprint_img,
        cmap=cmap_nprint,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )
    plt.xlabel("Header bits")
    plt.ylabel("Packets")
    plt.colorbar(ticks=[-1, 0, 1])
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 6. NPRINT IMAGE (figura 4)
    # -----------------------------
    C, H, W = nprint_img_compact.shape

    print("\n[NprintImage]")
    print("Shape:", nprint_img_compact.shape)
    print("Min:", nprint_img_compact.min(), "Max:", nprint_img_compact.max())

    # ---- 6.1 Visualización por canales ----
    titles = [
        "ch0: Presence",
        "ch1: Bit value",
        "ch2: Protocol group",
        "ch3: Variance"
    ]

    plt.figure(figsize=(15, 4))

    for i in range(C):
        plt.subplot(1, C, i + 1)
        plt.imshow(nprint_img_compact[i], aspect="auto")
        plt.title(titles[i] if i < len(titles) else f"ch{i}")
        plt.colorbar()
        plt.xlabel("Patches")
        plt.ylabel("Packets")

    plt.suptitle("NprintImage channels")
    plt.tight_layout()
    plt.show()

    # ---- 6.2 Visualización RGB compacta ----
    if C >= 3:
        rgb = nprint_img_compact[:3].transpose(1, 2, 0)

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title("NprintImage (pseudo-RGB)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---- 6.3 Métricas rápidas (muy útiles para análisis) ----
    print("\nStats:")
    print("Presence mean:", nprint_img_compact[0].mean())
    print("Bit mean:", nprint_img_compact[1].mean())
    if C > 3:
        print("Variance mean:", nprint_img_compact[3].mean())
    
    # -----------------------------
    # 7. MÉTRICAS CUANTITATIVAS
    # -----------------------------
    print("\n ===== COMPARATIVA CUANTITATIVA =====\n")

    representations = {
        "GASF": (gasf_img, original_size_flow),
        "Nprint": (nprint_img, original_size_window),
        "NprintImage": (nprint_img_compact, original_size_window)
    }

    for name, (rep, orig_size) in representations.items():
        rep_np = np.array(rep)

        sparsity = compute_sparsity(rep_np)
        entropy = compute_entropy(rep_np)
        comp_ratio = compression_ratio(orig_size, rep_np)

        print(f"--- {name} ---")
        print(f"Sparsity:        {sparsity:.4f}")
        print(f"Entropy:         {entropy:.4f}")
        print(f"Compression:     {comp_ratio:.4f}")
        print()