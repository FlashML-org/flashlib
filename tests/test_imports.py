"""Verify all top-level imports work and primitives are first-class."""
import flashlib


def test_top_level_primitives():
    """Every primitive should be importable directly from flashlib.*"""
    expected = [
        # algorithm primitives
        "flash_kmeans", "batch_kmeans_Euclid", "batch_kmeans_Cosine",
        "batch_kmeans_Dot", "kmeans_largeN", "kmeans_largeN_assign",
        "flash_knn",
        "flash_ivf_flat", "flash_ivf_flat_build", "flash_ivf_flat_search",
        "IvfFlatIndex",
        "flash_ivf_pq", "flash_ivf_pq_build", "flash_ivf_pq_search",
        "IvfPqIndex",
        "flash_pca",
        "flash_standard_scaler",
        "flash_dbscan",
        # linalg
        "cov_gemm", "eigh",
        # kernels
        "pairwise_l2", "pairwise_l2sq",
        # applications
        "KMeans", "FlashKMeans", "NearestNeighbors", "IVFFlat", "IVFPQ", "PCA",
        "StandardScaler", "DBSCAN",
        # diagnostics + info
        "diagnose", "info",
    ]
    missing = [name for name in expected if not hasattr(flashlib, name)]
    assert not missing, f"missing top-level names: {missing}"


def test_subpackage_paths():
    """Each primitive subpackage should be independently importable."""
    import flashlib.primitives.kmeans
    import flashlib.primitives.knn
    import flashlib.primitives.ivf_flat
    import flashlib.primitives.ivf_pq
    import flashlib.primitives.pca
    import flashlib.primitives.dbscan
    import flashlib.primitives.standard_scaler
    import flashlib.linalg.cov_gemm
    import flashlib.linalg.eigh
    import flashlib.kernels.distance


def test_diagnose():
    flashlib.diagnose()
