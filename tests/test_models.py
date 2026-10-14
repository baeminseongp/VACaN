import unittest

from vacan.models import (
    CongestionLevel,
    ObservationState,
    SensorType,
    congestion_level,
    observation_state,
)


class ObservationStateTests(unittest.TestCase):
    def test_unknown_is_distinct_from_fixed_coverage(self) -> None:
        self.assertEqual(observation_state(None, None, 30), ObservationState.UNKNOWN)
        self.assertEqual(
            observation_state(SensorType.CCTV, 0, 30), ObservationState.FIXED_COVERED
        )

    def test_robot_observation_becomes_stale(self) -> None:
        self.assertEqual(
            observation_state(SensorType.ROBOT, 15, 30), ObservationState.ROBOT_RECOVERED
        )
        self.assertEqual(observation_state(SensorType.ROBOT, 31, 30), ObservationState.STALE)


class CongestionLevelTests(unittest.TestCase):
    def test_missing_density_remains_unknown(self) -> None:
        self.assertEqual(congestion_level(None, 0.5, 1.5), CongestionLevel.UNKNOWN)

    def test_configurable_thresholds(self) -> None:
        self.assertEqual(congestion_level(0.2, 0.5, 1.5), CongestionLevel.LOW)
        self.assertEqual(congestion_level(0.5, 0.5, 1.5), CongestionLevel.MEDIUM)
        self.assertEqual(congestion_level(1.5, 0.5, 1.5), CongestionLevel.HIGH)


if __name__ == "__main__":
    unittest.main()
