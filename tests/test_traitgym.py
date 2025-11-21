"""Tests for TraitGym variant dataset loading."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from biofoundation.model.adapters.hf import HFTokenizer
from transformers import AutoTokenizer

from glm_experiments.data.traitgym import (
    download_genome,
    load_traitgym_dataset,
)


class TestHFTokenizer:
    """Tests for biofoundation HFTokenizer adapter."""

    @pytest.fixture
    def hf_tokenizer(self):
        """Load HuggingFace tokenizer for testing."""
        return AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")  # nosec B615

    def test_encode_returns_list_of_ints(self, hf_tokenizer):
        """Test that encode returns a list of integers."""
        adapter = HFTokenizer(hf_tokenizer)
        result = adapter.encode("ATGC")

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_produces_tokens(self, hf_tokenizer):
        """Test that encode produces tokens for DNA sequence."""
        adapter = HFTokenizer(hf_tokenizer)

        # Encode a simple sequence
        result = adapter.encode("ATGC")

        # Should produce tokens
        assert len(result) > 0

    def test_mask_token_id(self, hf_tokenizer):
        """Test that mask_token_id is accessible."""
        adapter = HFTokenizer(hf_tokenizer)

        assert hasattr(adapter, "mask_token_id")
        assert isinstance(adapter.mask_token_id, int)
        assert adapter.mask_token_id == hf_tokenizer.mask_token_id


class TestDownloadGenome:
    """Tests for download_genome function."""

    def test_download_genome_creates_file(self):
        """Test that download_genome creates a file at the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock URL and test file
            test_path = Path(tmpdir) / "test_genome.fa.gz"

            # Mock urlretrieve to create a dummy file
            with patch("urllib.request.urlretrieve") as mock_retrieve:

                def create_file(url, path):
                    Path(path).touch()

                mock_retrieve.side_effect = create_file

                result = download_genome("http://example.com/genome.fa.gz", test_path)

                assert result == test_path
                mock_retrieve.assert_called_once()

    def test_download_genome_skips_if_exists(self):
        """Test that download_genome skips download if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "existing_genome.fa.gz"
            test_path.touch()

            with patch("urllib.request.urlretrieve") as mock_retrieve:
                result = download_genome("http://example.com/genome.fa.gz", test_path)

                assert result == test_path
                mock_retrieve.assert_not_called()

    def test_download_genome_creates_parent_dirs(self):
        """Test that download_genome creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "nested" / "dir" / "genome.fa.gz"

            with patch("urllib.request.urlretrieve") as mock_retrieve:

                def create_file(url, path):
                    Path(path).touch()

                mock_retrieve.side_effect = create_file

                result = download_genome("http://example.com/genome.fa.gz", test_path)

                assert test_path.parent.exists()


class TestLoadTraitgymDataset:
    """Tests for load_traitgym_dataset function.

    Note: Tests that require the real genome file are marked with @pytest.mark.slow
    and require the genome to be downloaded first. These tests verify end-to-end
    functionality with real TraitGym data.

    Unit tests use mocks to test the data loading logic without the genome dependency.
    """

    @pytest.fixture
    def tokenizer_adapter(self):
        """Create a tokenizer adapter for testing."""
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "gonzalobenegas/tokenizer-dna-mlm"
        )  # nosec B615
        return HFTokenizer(hf_tokenizer)

    def test_load_traitgym_dataset_loads_raw_data(self, tokenizer_adapter):
        """Test that TraitGym dataset can be loaded from HuggingFace."""
        from datasets import load_dataset

        # Just test that the raw dataset loads correctly
        dataset = load_dataset("songlab/TraitGym", "mendelian_traits", split="test")  # nosec B615

        # Check expected columns exist
        assert "chrom" in dataset.column_names
        assert "pos" in dataset.column_names
        assert "ref" in dataset.column_names
        assert "alt" in dataset.column_names
        assert "label" in dataset.column_names

        # Check we have data
        assert len(dataset) > 0

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz").exists(),
        reason="Reference genome not downloaded. Run prepare_data() first.",
    )
    def test_load_traitgym_dataset_with_real_genome(self, tokenizer_adapter):
        """Test load_traitgym_dataset with real genome file.

        Requires the reference genome to be downloaded first.
        """
        from datasets import Dataset

        dataset = load_traitgym_dataset(
            tokenizer=tokenizer_adapter,
            genome_path="data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz",
            dataset_name="songlab/TraitGym",
            dataset_config="mendelian_traits",
            window_size=512,
        )

        assert isinstance(dataset, Dataset)

        # Check expected columns from transform_llr_mlm
        assert "input_ids" in dataset.column_names
        assert "pos" in dataset.column_names
        assert "ref" in dataset.column_names
        assert "alt" in dataset.column_names
        assert "label" in dataset.column_names

        # Get first example and verify format
        example = dataset[0]

        # Check input_ids has correct length
        # Note: HuggingFace datasets stores tensors as lists
        input_ids = example["input_ids"]
        assert len(input_ids) == 512

        # Check pos is an integer in valid range
        assert isinstance(example["pos"], int)
        assert example["pos"] == 256  # Center position for window_size=512

        # Check ref and alt are integers (token IDs)
        assert isinstance(example["ref"], int)
        assert isinstance(example["alt"], int)


class TestDNADataModuleWithTraitGym:
    """Tests for DNADataModule with TraitGym evaluation."""

    def test_datamodule_accepts_evals_config(self):
        """Test that DNADataModule accepts evals config parameter."""
        from glm_experiments.data.dna_datamodule import DNADataModule

        evals_config = {
            "traitgym": {
                "dataset_name": "songlab/TraitGym",
                "dataset_config": "mendelian_traits",
                "genome_url": "http://example.com/genome.fa.gz",
                "genome_path": "data/genome.fa.gz",
                "window_size": 512,
                "batch_size": 128,
            }
        }

        dm = DNADataModule(evals=evals_config)

        assert dm.hparams.evals == evals_config

    def test_datamodule_evals_none_by_default(self):
        """Test that evals is None by default."""
        from glm_experiments.data.dna_datamodule import DNADataModule

        dm = DNADataModule()

        assert dm.hparams.evals is None
        assert dm.data_val_traitgym is None

    def test_val_dataloader_returns_single_loader_without_evals(self):
        """Test that val_dataloader returns single DataLoader without evals."""
        from torch.utils.data import DataLoader

        from glm_experiments.data.dna_datamodule import DNADataModule

        dm = DNADataModule(
            batch_size=32,
            num_workers=0,
            evals=None,
        )

        # Mock the data_val attribute
        dm.data_val = [{"input_ids": torch.zeros(10), "labels": torch.zeros(10)}]
        dm.data_val_traitgym = None
        dm.batch_size_per_device = 32

        result = dm.val_dataloader()

        assert isinstance(result, DataLoader)

    def test_val_dataloader_returns_list_with_evals(self):
        """Test that val_dataloader returns list of DataLoaders with evals."""
        from torch.utils.data import DataLoader

        from glm_experiments.data.dna_datamodule import DNADataModule

        evals_config = {
            "traitgym": {
                "batch_size": 128,
            }
        }

        dm = DNADataModule(
            batch_size=32,
            num_workers=0,
            evals=evals_config,
        )

        # Mock the data attributes
        dm.data_val = [{"input_ids": torch.zeros(10), "labels": torch.zeros(10)}]
        dm.data_val_traitgym = [
            {"input_ids": torch.zeros(512), "pos": 256, "ref": 1, "alt": 2, "label": 0}
        ]
        dm.batch_size_per_device = 32

        result = dm.val_dataloader()

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(dl, DataLoader) for dl in result)

    def test_hydra_instantiation_with_evals(self):
        """Test that DataModule can be instantiated from Hydra config with evals."""
        from hydra import compose, initialize

        from glm_experiments.data.dna_datamodule import DNADataModule

        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="train.yaml",
                overrides=["data=default"],
            )

            # Instantiate datamodule
            import hydra

            dm = hydra.utils.instantiate(cfg.data)

            # Check that it's the right type
            assert isinstance(dm, DNADataModule)

            # Check evals config is present
            assert dm.hparams.evals is not None
            assert "traitgym" in dm.hparams.evals
            assert dm.hparams.evals["traitgym"]["dataset_name"] == "songlab/TraitGym"
            assert dm.hparams.evals["traitgym"]["window_size"] == 512
