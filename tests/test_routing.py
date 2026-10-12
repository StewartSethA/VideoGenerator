import unittest
from videogenerator.routing import choose_backend, capabilities, candidates


class RoutingTests(unittest.TestCase):
    def test_catalog_unifies_builtins_and_ecosystem_without_overclaiming(self):
        rows = capabilities()
        by_id = {row["id"]: row for row in rows}
        self.assertTrue(by_id["sdxl"]["executable"])
        self.assertTrue(by_id["wan"]["executable"])
        self.assertFalse(by_id["wan22_t2v_a14b"]["executable"])
        self.assertEqual(by_id["wan22_i2v_a14b"]["inputs"], ["image"])
        self.assertTrue(by_id["ltx2"]["keyframes"])

    def test_router_prefers_executable_backend(self):
        self.assertEqual(choose_backend("image"), "sdxl")
        self.assertIn(choose_backend("video"), {"wan", "ltx", "cogvideox"})
        self.assertEqual(choose_backend("video", input_kinds=["image"]), "ltx")

    def test_router_uses_executable_ltx2_for_multiple_keyframes(self):
        result = candidates("video", input_kinds=["image"], keyframes=2)
        self.assertTrue(any(not item["executable"] for item in result))
        self.assertEqual(
            choose_backend("video", input_kinds=["image"], keyframes=2),
            "ltx2",
        )

    def test_router_can_include_external_candidates_explicitly(self):
        self.assertEqual(
            choose_backend("video", input_kinds=["image"], keyframes=2, require_executable=False),
            "ltx2",
        )


if __name__ == "__main__":
    unittest.main()
