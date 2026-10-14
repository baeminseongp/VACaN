import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "apps/web/source-data/dashboard.json"


class WebExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(BUNDLE.read_text(encoding="utf-8"))

    def test_expected_linked_view_dimensions(self):
        self.assertEqual(70, len(self.payload["zones"]))
        self.assertEqual(654, len(self.payload["frames"]))
        self.assertEqual(3, len(self.payload["robots"]))
        self.assertTrue(self.payload["events"])

    def test_frames_align_to_zone_order(self):
        zone_count = len(self.payload["zoneIds"])
        for frame in self.payload["frames"]:
            self.assertEqual(zone_count, len(frame["g"]))
            self.assertEqual(zone_count, len(frame["o"]))
            self.assertEqual(5, len(frame["c"]))
            self.assertEqual(2, len(frame["d"]))
            self.assertLessEqual(frame["d"][0], max(frame["g"]))
            self.assertLessEqual(frame["d"][1], max(frame["g"]))

    def test_bundle_excludes_individual_trajectory_fields(self):
        serialized = json.dumps(self.payload).lower()
        for forbidden in ("person_id", "track_id", "trajectory_id", "position_x", "position_y"):
            self.assertNotIn(forbidden, serialized)

    def test_each_planning_pair_has_two_strategies(self):
        for pair in self.payload["planning"].values():
            self.assertEqual({"BASELINE", "CONGESTION_AWARE"}, set(pair))

    def test_caviar_grid_has_local_aggregate_evidence(self):
        caviar = self.payload["caviar"]
        self.assertEqual(30, caviar["zoneCount"])
        self.assertEqual(2360, caviar["frameCount"])
        self.assertEqual(25, caviar["fps"])
        self.assertTrue(caviar["localOnly"])
        self.assertEqual(2, len(caviar["cameras"]))
        self.assertTrue(any(zone["observations"] for zone in caviar["zones"]))
        self.assertEqual("DEMO_ONLY_NOT_SPATIALLY_ALIGNED", caviar["mappingMode"])
        self.assertEqual(70, len(caviar["demoMappings"]))
        mapped_cameras = {mapping["camera"] for mapping in caviar["demoMappings"].values()}
        self.assertEqual({0, 1}, mapped_cameras)
        self.assertGreaterEqual(
            len({mapping["frame"] for mapping in caviar["demoMappings"].values()}),
            30,
        )

    def test_caviar_export_does_not_disclose_raw_media_paths(self):
        serialized = json.dumps(self.payload["caviar"]).lower()
        self.assertNotIn("/media/engineer", serialized)
        self.assertNotIn(".mpg", serialized)

    def test_demo_cctv_is_integrated_into_atc_inspector(self):
        html = (ROOT / "apps/web/static/index.html").read_text(encoding="utf-8")
        script = (ROOT / "apps/web/static/app.js").read_text(encoding="utf-8")
        self.assertIn('class="demo-cctv"', html)
        self.assertIn('id="evidenceVideo"', html)
        self.assertIn("NOT SPATIALLY ALIGNED", html)
        self.assertNotIn('id="caviarDialog"', html)
        self.assertNotIn("caviar-panel", html)
        self.assertNotIn('data-dataset="caviar"', html)
        self.assertIn('fetch(source, { method: "HEAD"', script)
        self.assertNotIn('includes(window.location.hostname)', script)

    def test_built_assets_are_cache_versioned(self):
        html = (ROOT / "apps/web/dist/index.html").read_text(encoding="utf-8")
        self.assertNotIn("__BUILD_VERSION__", html)
        self.assertRegex(html, r"styles\.css\?v=[0-9a-f]{12}")
        self.assertRegex(html, r"app\.js\?v=[0-9a-f]{12}")

    def test_inspector_scroll_does_not_stretch_map(self):
        styles = (ROOT / "apps/web/static/styles.css").read_text(encoding="utf-8")
        self.assertIn(".map-panel { align-self: start; }", styles)
        self.assertIn("height: 571px", styles)
        self.assertIn("#atcInspector { flex: 1; min-height: 0; overflow-y: auto;", styles)


if __name__ == "__main__":
    unittest.main()
