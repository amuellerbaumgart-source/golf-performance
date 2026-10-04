import os
from pathlib import Path
import tempfile

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"
os.environ.setdefault("GOLF_PERFORMANCE_DATA_DIR", tempfile.mkdtemp(prefix="golf-app-test-"))


def test_app_supports_multiple_metrics_and_primary_metric_selection() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    app.get_by_key("selected_metrics").set_value(["Carry", "Offline distance"]).run()
    assert not app.exception

    app.get_by_key("primary_metric").select("Offline distance").run()
    assert not app.exception
    assert any("Primary metric" in expander.label for expander in app.expander)


def test_app_renders_definition_screen_without_exception() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any("Define your experiment" in header.value for header in app.header)
    assert app.get_by_key("generate_protocol_options")


def test_app_generates_protocol_options_from_valid_form() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    app.get_by_key("experiment_name").set_value("Driver loft test")
    app.get_by_key("changed_variable").set_value("Driver loft")
    app.get_by_key("baseline_value").set_value("9 degrees")
    app.get_by_key("treatment_value").set_value("10 degrees")
    app.get_by_key("primary_goal").set_value("Increase carry")
    app.get_by_key("generate_protocol_options").click().run()

    assert not app.exception
    assert any("Choose your testing plan" in header.value for header in app.header)
    assert app.get_by_key("select_confirmatory")
    assert app.get_by_key("select_exploratory")
    assert any("operationally demanding" in warning.value for warning in app.warning)


def test_app_can_select_exploratory_protocol() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    app.get_by_key("experiment_name").set_value("Driver loft test")
    app.get_by_key("changed_variable").set_value("Driver loft")
    app.get_by_key("baseline_value").set_value("9 degrees")
    app.get_by_key("treatment_value").set_value("10 degrees")
    app.get_by_key("primary_goal").set_value("Increase carry")
    app.get_by_key("generate_protocol_options").click().run()
    app.get_by_key("select_exploratory").click().run()

    assert not app.exception
    assert any("Selected plan: Exploratory" in success.value for success in app.success)
    assert app.dataframe


def test_app_can_open_shot_entry_after_selecting_a_protocol() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    for key, value in {
        "experiment_name": "Driver loft test",
        "changed_variable": "Driver loft",
        "baseline_value": "9 degrees",
        "treatment_value": "10 degrees",
        "primary_goal": "Increase carry",
    }.items():
        app.get_by_key(key).set_value(value)
    app.get_by_key("exploratory_cap").set_value(10)
    app.get_by_key("generate_protocol_options").click().run()
    app.get_by_key("select_exploratory").click().run()
    app.get_by_key("continue_to_data_entry").click().run()

    assert not app.exception
    assert any("Enter shot results" in header.value for header in app.header)
