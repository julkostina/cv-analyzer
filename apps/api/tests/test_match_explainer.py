"""Tests for SHAP/LIME match explainability."""

from unittest.mock import patch

from app.services.match_explainer import (
    _build_features,
    component_attributions,
    explain_match_score,
)
from app.services.semantic_matcher import SemanticMatchResult


def test_component_attributions_weighted_sum():
    sem = SemanticMatchResult(
        score=0.72,
        skills_similarity=0.8,
        experience_similarity=0.6,
        overall_similarity=0.7,
    )
    attrs = component_attributions(sem)
    assert attrs["skills"] == round(0.5 * 0.8, 6)
    assert attrs["experience"] == round(0.3 * 0.6, 6)
    assert attrs["overall"] == round(0.2 * 0.7, 6)
    assert abs(sum(attrs.values()) - sem.score) < 1e-6


def test_build_features_caps_and_labels():
    features = _build_features(
        skills=["Python", "React", "Go"],
        experience=None,
        cv_experience_text="Senior Dev — Acme — 2020-2024",
    )
    assert features.names[0] == "Навичка: Python"
    assert any(n.startswith("Досвід:") for n in features.names)


def test_explain_match_score_with_mocked_semantic_match():
    sem = SemanticMatchResult(
        score=0.55,
        skills_similarity=0.6,
        experience_similarity=0.5,
        overall_similarity=0.5,
    )

    def fake_compute(cv_skills, cv_experience_text, cv_full_text, job_requirements_text, job_full_text):
        n = len(cv_skills) + (1 if cv_experience_text.strip() else 0)
        base = 0.1 * n
        return SemanticMatchResult(
            score=min(1.0, base),
            skills_similarity=base,
            experience_similarity=base,
            overall_similarity=0.5,
        )

    with patch("app.services.match_explainer.compute_semantic_match", side_effect=fake_compute):
        result = explain_match_score(
            sem=sem,
            skills=["Python", "SQL"],
            experience=None,
            cv_experience_text="Dev at Foo",
            cv_full_text="CV text",
            job_requirements_text="Job",
            job_full_text="Job",
        )

    assert result is not None
    assert result.component_attributions is not None
    assert result.shap is not None or result.lime is not None
