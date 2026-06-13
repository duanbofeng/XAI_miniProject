from pathlib import Path

from xai_miniproject.config import load_config


def test_load_aifb_config_resolves_paths() -> None:
    config = load_config("configs/aifb.yaml")
    assert config.dataset.name == "aifb"
    assert config.dataset.rdf_path.is_absolute()
    assert config.dataset.rdf_path.name == "aifbfixed_complete.n3"
    assert config.project.artifacts_dir == Path("artifacts/aifb").resolve()
    assert "http://swrc.ontoware.org/ontology#affiliation" in config.dataset.exclude_predicates
