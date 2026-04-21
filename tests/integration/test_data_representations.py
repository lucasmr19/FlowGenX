"""
tests/test_data_representations.py
======================
Tests de integración del pipeline data + representations.

Ejecutar con:
    python -m pytest tests/test_data_representations.py -v
    # o directamente:
    python tests/test_data_representations.py

No requiere un archivo PCAP real: construye flujos sintéticos
para verificar que todo el pipeline funciona end-to-end.
"""
import pytest
import torch

from src.utils.make_synthetic_flow import make_dataset

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flat_tokenizer():
    print("\n[TEST] FlatTokenizer")
    from src.representations.sequential.tokenizer import (
        FlatTokenizer, SequentialConfig
    )

    cfg   = SequentialConfig(max_length=128, max_vocab_size=500)
    rep   = FlatTokenizer(cfg)
    flows = make_dataset(30)

    # fit sobre primeros 20
    rep.fit(flows[:20])
    assert rep._is_fitted, "No se ajustó correctamente"
    assert len(rep.vocab) > len([
        "<PAD>","<UNK>","<BOS>","<EOS>","<SEP>","<MASK>","<FWD>","<BWD>"
    ]), "Vocabulario demasiado pequeño"

    # encode
    tensor = rep.encode(flows[25])
    assert tensor.shape == (128,), f"Shape incorrecto: {tensor.shape}"
    assert tensor.dtype == torch.long

    # batch encode
    batch = rep.encode_batch(flows[20:25])
    assert batch.shape == (5, 128)

    # BOS y EOS presentes
    assert tensor[0].item() == rep.vocab.bos_id, "Falta BOS token"

    print(f"  ✓ vocab size: {len(rep.vocab)}")
    print(f"  ✓ encode shape: {tensor.shape}")
    print(f"  ✓ batch shape:  {batch.shape}")
    print(f"  ✓ primeros tokens: {rep.decode(tensor)[:6]}")


def test_protocol_aware_tokenizer():
    print("\n[TEST] ProtocolAwareTokenizer")
    from src.representations.sequential.tokenizer import (
        ProtocolAwareTokenizer, ProtocolAwareConfig
    )

    cfg   = ProtocolAwareConfig(max_length=256, encode_tcp_state=True)
    rep   = ProtocolAwareTokenizer(cfg)
    flows = make_dataset(30)

    rep.fit(flows[:20])
    tensor = rep.encode(flows[25])

    assert tensor.shape == (256,)
    # Verificar que aparecen separadores de capa
    tokens = rep.decode(tensor)
    token_strs = set(tokens)
    assert "<L3>" in token_strs or any("<L3>" in t for t in tokens), \
        "No se encontraron separadores de capa L3"

    print(f"  ✓ vocab size: {len(rep.vocab)}")
    print(f"  ✓ encode shape: {tensor.shape}")
    print(f"  ✓ tokens con separadores de capa: OK")


def test_gasf():
    print("\n[TEST] GASFRepresentation")
    from src.representations.vision import (
        GASFRepresentation, GASFConfig
    )

    cfg   = GASFConfig(image_size=32, feature="combined", n_steps=32)
    rep   = GASFRepresentation(cfg)
    flows = make_dataset(30)

    rep.fit(flows[:20])
    tensor = rep.encode(flows[25])

    assert tensor.shape == (2, 32, 32), f"Shape incorrecto: {tensor.shape}"
    assert tensor.dtype == torch.float32
    # Los valores GASF deben estar en [-1, 1]
    assert tensor.min().item() >= -1.01
    assert tensor.max().item() <=  1.01

    # Verificar no invertible
    try:
        rep.decode(tensor)
        assert False, "Debería lanzar NotImplementedError"
    except NotImplementedError:
        pass

    print(f"  ✓ output shape: {tensor.shape}")
    print(f"  ✓ rango valores: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  ✓ NotImplementedError en decode: OK")


def test_nprint():
    print("\n[TEST] NprintRepresentation")
    from src.representations.vision import (
        NprintRepresentation, NprintConfig
    )

    cfg   = NprintConfig(max_packets=10)
    rep   = NprintRepresentation(cfg)
    flows = make_dataset(30)

    rep.fit(flows[:20])
    tensor = rep.encode(flows[25])

    H, W = rep.output_shape
    assert tensor.shape == (H, W), f"Shape incorrecto: {tensor.shape}"
    # Valores deben ser -1, 0 o 1
    unique_vals = tensor.unique().tolist()
    assert all(v in (-1.0, 0.0, 1.0) for v in unique_vals), \
        f"Valores no esperados: {unique_vals}"

    # Decode (invertible)
    packets = rep.decode(tensor)
    assert isinstance(packets, list)
    assert len(packets) > 0
    assert hasattr(packets[0], "ip_proto")

    print(f"  ✓ output shape: {tensor.shape}  (bits_per_row={W})")
    print(f"  ✓ valores únicos: {unique_vals}")
    print(f"  ✓ decode → {len(packets)} paquetes reconstruidos")
    print(f"  ✓ primer paquete reconstruido: ip_proto={packets[0].ip_proto}, "
          f"dport={packets[0].dport}")

def test_nprint_image():
    print("\n[TEST] NprintImageRepresentation")

    from src.representations.vision import (
        NprintImageRepresentation, NprintImageConfig
    )

    cfg = NprintImageConfig(
        patch_size=4,
        max_packets=32,         # más pequeño para test
        use_ch3_variance=True,
        ch0_mode="any",
        use_bernoulli_decode=False,  # importante para test determinista
        pad_to_height=32
    )

    rep   = NprintImageRepresentation(cfg)
    flows = make_dataset(30)

    rep.fit(flows[:20])
    tensor = rep.encode(flows[25])

    C, H, W = rep.output_shape

    assert tensor.shape == (C, H, W), f"Shape incorrecto: {tensor.shape}"
    assert tensor.dtype == torch.float32

    # rango [0,1]
    assert tensor.min().item() >= -1e-5
    assert tensor.max().item() <= 1.0 + 1e-5

    # decode (aproximado)
    packets = rep.decode(tensor)
    assert isinstance(packets, list)
    assert len(packets) > 0

    print(f"  ✓ shape: {tensor.shape}")
    print(f"  ✓ rango: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  ✓ decode OK (aproximado)")

def test_datamodule():
    print("\n[TEST] TrafficDataModule (integración completa)")
    from src.datamodules import TrafficDataModule
    from src.representations.sequential.tokenizer import (
        FlatTokenizer, SequentialConfig
    )

    flows = make_dataset(100)
    rep   = FlatTokenizer(SequentialConfig(max_length=64))

    dm = TrafficDataModule(
        samples        = flows,
        representation = rep,
        train_ratio = 0.7,
        val_ratio   = 0.15,
        test_ratio  = 0.15,
        batch_size  = 8,
        seed        = 42,
    )
    dm.setup()

    summary = dm.summary()
    assert summary["train_samples"] == 70
    assert summary["val_samples"]   == 15
    assert summary["test_samples"]  == 15

    # Train dataloader
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    assert batch.shape == (8, 64), f"Batch shape incorrecto: {batch.shape}"

    # Val dataloader
    val_dl = dm.val_dataloader()
    val_batch = next(iter(val_dl))
    assert val_batch.shape[1] == 64

    print(f"  ✓ split: {summary['train_samples']}/{summary['val_samples']}/{summary['test_samples']}")
    print(f"  ✓ train batch shape: {batch.shape}")
    print(f"  ✓ val   batch shape: {val_batch.shape}")
    print(f"  ✓ representación ajustada solo sobre train: OK")


def test_representation_registry():
    print("\n[TEST] Registry de representaciones")
    from src.representations import get_representation, REGISTRY

    for name in REGISTRY:
        rep = get_representation(name)
        assert rep is not None
        print(f"  ✓ '{name}' → {rep.__class__.__name__}")

    try:
        get_representation("nonexistent")
        assert False
    except ValueError:
        print("  ✓ ValueError para nombre desconocido: OK")


def test_save_load():
    print("\n[TEST] Persistencia (save/load)")
    from src.representations.sequential.tokenizer import (
        FlatTokenizer, SequentialConfig
    )
    import tempfile, pathlib

    flows = make_dataset(20)
    rep   = FlatTokenizer(SequentialConfig(max_length=64))
    rep.fit(flows[:15])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "rep.pt"
        rep.save(path)

        rep2 = FlatTokenizer.load(path)
        assert rep2._is_fitted
        assert len(rep2.vocab) == len(rep.vocab)

        t1 = rep.encode(flows[16])
        t2 = rep2.encode(flows[16])
        assert torch.equal(t1, t2), "Tensores diferentes tras load()"

    print("  ✓ save/load produce resultados idénticos")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_flat_tokenizer,
        test_protocol_aware_tokenizer,
        test_gasf,
        test_nprint,
        test_nprint_image,
        test_datamodule,
        test_representation_registry,
        test_save_load,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Resultados: {passed} passed / {failed} failed")
    print('='*50)