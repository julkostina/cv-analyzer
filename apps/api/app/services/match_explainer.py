"""SHAP and LIME local explanations for embedding-based match_score."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.models.cv_models import (
    ExperienceItem,
    ExplainabilityAttribution,
    ExplainabilityMethodResult,
    MatchExplainability,
)
from app.services.semantic_matcher import SemanticMatchResult, compute_semantic_match, normalized_semantic_weights

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExplainFeatures:
    names: Tuple[str, ...]
    skills: Tuple[str, ...]
    experience_lines: Tuple[str, ...]


def _experience_lines(
    experience: Optional[Sequence[ExperienceItem]],
    fallback_text: str,
) -> List[str]:
    lines: List[str] = []
    if experience:
        for item in experience:
            parts = [p for p in (item.title, item.employer, item.duration) if p and str(p).strip()]
            if parts:
                lines.append(" — ".join(str(p).strip() for p in parts))
    if not lines and fallback_text.strip():
        for chunk in fallback_text.replace("•", "\n").split("\n"):
            c = chunk.strip()
            if c:
                lines.append(c[:400])
    return lines


def _build_features(
    skills: List[str],
    experience: Optional[Sequence[ExperienceItem]],
    cv_experience_text: str,
) -> _ExplainFeatures:
    skill_feats = [s.strip() for s in skills if s and s.strip()][: settings.explainer_max_skills]
    exp_lines = _experience_lines(experience, cv_experience_text)[: settings.explainer_max_experience]
    names: List[str] = [f"Навичка: {s}" for s in skill_feats]
    names.extend(f"Досвід: {line}" for line in exp_lines)
    return _ExplainFeatures(names=tuple(names), skills=tuple(skill_feats), experience_lines=tuple(exp_lines))


def _make_predictor(
    features: _ExplainFeatures,
    cv_full_text: str,
    job_requirements_text: str,
    job_full_text: str,
):
    def predict(masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        scores = np.zeros(masks.shape[0], dtype=float)
        n_skills = len(features.skills)
        for i, row in enumerate(masks):
            active_skills = [s for s, on in zip(features.skills, row[:n_skills]) if on >= 0.5]
            active_exp = [e for e, on in zip(features.experience_lines, row[n_skills:]) if on >= 0.5]
            sem = compute_semantic_match(
                cv_skills=list(active_skills),
                cv_experience_text="\n".join(active_exp),
                cv_full_text=cv_full_text,
                job_requirements_text=job_requirements_text,
                job_full_text=job_full_text,
            )
            scores[i] = sem.score
        return scores

    return predict


def component_attributions(sem: SemanticMatchResult) -> Dict[str, float]:
    ws, we, wo = normalized_semantic_weights()
    return {
        "skills": round(ws * sem.skills_similarity, 6),
        "experience": round(we * sem.experience_similarity, 6),
        "overall": round(wo * sem.overall_similarity, 6),
    }


def _split_attributions(
    values: np.ndarray,
    names: Sequence[str],
    *,
    top_k: int,
) -> Tuple[List[ExplainabilityAttribution], List[ExplainabilityAttribution]]:
    pairs = [(names[i], float(values[i])) for i in range(len(names))]
    positive = sorted((p for p in pairs if p[1] > 0), key=lambda x: x[1], reverse=True)[:top_k]
    negative = sorted((p for p in pairs if p[1] < 0), key=lambda x: x[1])[:top_k]
    return (
        [ExplainabilityAttribution(feature=f, contribution=c) for f, c in positive],
        [ExplainabilityAttribution(feature=f, contribution=c) for f, c in negative],
    )


def _run_shap(
    predict,
    features: _ExplainFeatures,
    baseline_score: float,
    predicted_score: float,
) -> Optional[ExplainabilityMethodResult]:
    try:
        import shap
    except ImportError:
        logger.warning("shap package not installed; skipping SHAP explainability")
        return None

    n = len(features.names)
    if n == 0:
        return None

    instance = np.ones(n, dtype=float)
    background = np.zeros((1, n), dtype=float)
    try:
        explainer = shap.KernelExplainer(predict, background, link="identity")
        raw = explainer.shap_values(instance.reshape(1, -1), nsamples=settings.explainer_shap_samples)
        values = np.asarray(raw, dtype=float).reshape(-1)[:n]
    except Exception:
        logger.warning("SHAP KernelExplainer failed", exc_info=True)
        return None

    top_pos, top_neg = _split_attributions(values, features.names, top_k=settings.explainer_top_features)
    return ExplainabilityMethodResult(
        method="shap",
        baseline_score=round(baseline_score, 6),
        predicted_score=round(predicted_score, 6),
        top_positive=top_pos,
        top_negative=top_neg,
    )


def _run_lime(
    predict,
    features: _ExplainFeatures,
    baseline_score: float,
    predicted_score: float,
) -> Optional[ExplainabilityMethodResult]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("lime package not installed; skipping LIME explainability")
        return None

    n = len(features.names)
    if n == 0:
        return None

    rng = np.random.default_rng(42)
    training = rng.integers(0, 2, size=(max(32, settings.explainer_lime_samples), n)).astype(float)
    instance = np.ones(n, dtype=float)

    try:
        explainer = LimeTabularExplainer(
            training_data=training,
            feature_names=list(features.names),
            mode="regression",
            discretize_continuous=False,
        )
        explanation = explainer.explain_instance(
            instance,
            predict,
            num_features=min(n, settings.explainer_top_features),
            num_samples=settings.explainer_lime_samples,
        )
        values = np.zeros(n, dtype=float)
        for idx, weight in explanation.as_map().get(1, []):
            values[idx] = float(weight)
    except Exception:
        logger.warning("LIME TabularExplainer failed", exc_info=True)
        return None

    top_pos, top_neg = _split_attributions(values, features.names, top_k=settings.explainer_top_features)
    return ExplainabilityMethodResult(
        method="lime",
        baseline_score=round(baseline_score, 6),
        predicted_score=round(predicted_score, 6),
        top_positive=top_pos,
        top_negative=top_neg,
    )


def explain_match_score(
    *,
    sem: SemanticMatchResult,
    skills: List[str],
    experience: Optional[Sequence[ExperienceItem]],
    cv_experience_text: str,
    cv_full_text: str,
    job_requirements_text: str,
    job_full_text: str,
) -> Optional[MatchExplainability]:
    if not settings.use_match_explainers:
        return None

    features = _build_features(skills, experience, cv_experience_text)
    comps = component_attributions(sem)

    if not features.names:
        return MatchExplainability(
            component_attributions=comps,
            shap=None,
            lime=None,
        )

    predict = _make_predictor(features, cv_full_text, job_requirements_text, job_full_text)
    n = len(features.names)
    baseline_score = float(predict(np.zeros(n))[0])
    predicted_score = float(predict(np.ones(n))[0])

    shap_result = _run_shap(predict, features, baseline_score, predicted_score)
    lime_result = _run_lime(predict, features, baseline_score, predicted_score)

    if shap_result is None and lime_result is None:
        return MatchExplainability(
            component_attributions=comps,
            shap=None,
            lime=None,
        )

    return MatchExplainability(
        component_attributions=comps,
        shap=shap_result,
        lime=lime_result,
    )
