from scarcity.fmi.validator import FMIValidator, ValidatorConfig


def test_fmi_validator_requires_dp_when_configured():
    validator = FMIValidator(config=ValidatorConfig(dp_required=True))
    payload = {
        "type": "msp",
        "schema_hash": "x",
        "rev": 3,
        "domain_id": "d",
        "profile_class": "p",
        "metrics": {},
        "controller": {},
        "evaluator": {},
        "operators": {},
    }
    result = validator.validate(payload)
    assert not result.ok and result.reason in {"dp_flag_missing", "dp_params_missing"}
