import unittest
from pathlib import Path

from vacan.datasets.caviar.annotations import iter_annotations, xml_frame_count
from vacan.datasets.caviar.homography import fit_homography, rmse


class CaviarAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures/caviar/minimal.xml"

    def test_streams_boxes_and_uses_bottom_center(self):
        annotations = list(iter_annotations(self.fixture))
        self.assertEqual(1, len(annotations))
        self.assertEqual((40.0, 60.0), annotations[0].bottom_center)
        self.assertEqual(2, xml_frame_count(self.fixture))

    def test_homography_maps_four_control_points(self):
        source = [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)]
        target = [(10.0, 20.0), (14.0, 20.0), (10.0, 23.0), (14.0, 23.0)]
        transform = fit_homography(source, target)
        projected = [transform.project(point) for point in source]
        self.assertLess(rmse(projected, target), 1e-10)
        self.assertAlmostEqual(12.0, transform.project((1.0, 0.5))[0])
        self.assertAlmostEqual(21.5, transform.project((1.0, 0.5))[1])

    def test_degenerate_correspondences_are_rejected(self):
        collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        with self.assertRaises(ValueError):
            fit_homography(collinear, collinear)


if __name__ == "__main__":
    unittest.main()
