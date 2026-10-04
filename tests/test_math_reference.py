from pathlib import Path


MATH_PATH = Path(__file__).parents[1] / "MATH.txt"


def test_math_reference_documents_current_method_and_planned_correction() -> None:
    contents = MATH_PATH.read_text(encoding="utf-8")

    for required_text in (
        "welch_independent_two_sample_t",
        "individual_shot",
        "Welch-Satterthwaite",
        "Hedges' g",
        "paired block analysis",
        "estimated_shot_SD * sqrt(2 / shots_per_block)",
        "Bootstrap sensitivity analysis",
    ):
        assert required_text in contents
