from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/v2_data_requirements.yaml"
DOC_PATH = PROJECT_ROOT / "docs/v2_data_gap_analysis.md"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v2_plan_is_non_executing_and_preserves_v1() -> None:
    config = load_config()
    assert config["status"] == "PLAN_ONLY"
    assert config["created_from_existing_artifacts_only"] is True
    assert all(config["prohibited_actions"].values())


def test_stage18_locked_test_is_sealed_and_forbidden_for_selection() -> None:
    config = load_config()
    stage18 = config["evidence_baseline"]["stage18"]
    assert stage18["status"] == "SEALED_DIAGNOSTIC_REFERENCE_ONLY"
    assert {"threshold_selection", "model_selection", "retraining_feedback_loop"} <= set(
        stage18["prohibited_uses"]
    )
    assert "V1_STAGE18_LOCKED_TEST" in config["experiments"]["forbidden_model_selection_data"]


def test_source_and_application_taxonomies_remain_separate() -> None:
    taxonomy = load_config()["taxonomy"]
    assert taxonomy["preserve_source_taxonomy"] is True
    assert taxonomy["preserve_application_mapping_separately"] is True
    assert set(taxonomy["application_classes"]) == {
        "bicycle", "bus", "car", "person", "motorcycle", "truck"
    }
    assert taxonomy["proposed_annotation_policy"]["rider"]["application_class"] == "person"


def test_data_acceptance_requires_provenance_license_coverage_and_leakage_controls() -> None:
    criteria = load_config()["acceptance_criteria"]
    assert criteria["provenance"]["immutable_archive_hash_required"] is True
    assert criteria["license"]["explicit_license_required"] is True
    assert criteria["annotations"]["person_completeness_audit_required"] is True
    assert criteria["coverage_gate"]["acceptance_not_based_on_image_count_alone"] is True
    assert criteria["grouping_and_leakage"]["group_cross_split_allowed"] is False
    assert criteria["grouping_and_leakage"]["v1_stage18_locked_test_overlap_allowed"] is False


def test_experiment_priority_is_fixed_and_escalates_model_size_last() -> None:
    experiments = load_config()["experiments"]
    ordered = experiments["fixed_order"]
    assert [item["priority"] for item in ordered] == [1, 2, 3, 4, 5]
    assert ordered[1]["model"] == "yolo26n.pt" and ordered[1]["imgsz"] == 640
    assert ordered[2]["model"] == "yolo26n.pt" and ordered[2]["imgsz"] == 960
    assert ordered[3]["model"] == "yolo26s.pt"
    assert "Only if" in ordered[3]["condition"]
    assert ordered[4]["selection_role"] == "FINAL_EVALUATION_ONLY"
    assert experiments["model_selection_data"] == ["V2_TRAIN", "V2_VAL"]


def test_new_v2_holdout_is_sealed_and_single_use() -> None:
    governance = load_config()["v2_holdout_governance"]
    assert governance["status_before_final_evaluation"] == "SEALED"
    assert "single_final_evaluation" in governance["allowed_uses"]
    assert {"training", "threshold_tuning", "model_selection"} <= set(governance["prohibited_uses"])
    assert "V1_STAGE18_LOCKED_TEST" in governance["must_not_overlap"]


def test_document_separates_evidence_hypotheses_and_experiments() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")
    assert "Observed Evidence" in document
    assert "Hypothesis" in document
    assert "Planned Experiment" in document
    assert "did not download data" in document
    assert "did not" in document and "train a model" in document
