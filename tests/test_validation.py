import unittest
from pathlib import Path

from vacan.validation import validate_scenario


class ScenarioValidationTests(unittest.TestCase):
    def test_minimal_fixture_satisfies_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = validate_scenario(root / "data" / "synthetic" / "minimal")
        self.assertEqual(result.errors, ())


if __name__ == "__main__":
    unittest.main()
