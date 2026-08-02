import sys
import unittest

import numpy as np

from shapez_rl.encoding import MapEncoder

BOUNDS = {"x": -4, "y": -4, "w": 8, "h": 8}


def building(building_id, x, y, w=1, h=1, rotation=0):
    return {
        "id": building_id,
        "x": x,
        "y": y,
        "rotation": rotation,
        "bounds": {"x": x, "y": y, "w": w, "h": h},
    }


class MapEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = MapEncoder(BOUNDS)
        self.channel = {name: i for i, name in enumerate(self.encoder.channel_names)}

    def test_shape_matches_bounds(self):
        self.assertEqual(self.encoder.shape, (self.encoder.num_channels, 8, 8))

    def test_empty_map_is_all_zero(self):
        grid = self.encoder.encode({"resources": [], "buildings": []})
        self.assertEqual(grid.shape, self.encoder.shape)
        self.assertEqual(grid.sum(), 0.0)

    def test_resource_sets_generic_and_specific_channels(self):
        payload = {
            "resources": [{"x": -4, "y": -4, "type": "shape", "key": "CuCuCuCu"}],
            "buildings": [],
        }
        grid = self.encoder.encode(payload)
        self.assertEqual(grid[self.channel["resource_shape_any"], 0, 0], 1.0)
        self.assertEqual(grid[self.channel["resource_shape_CuCuCuCu"], 0, 0], 1.0)
        self.assertEqual(grid[self.channel["resource_color_any"], 0, 0], 0.0)

    def test_unknown_resource_key_still_sets_generic_channel(self):
        payload = {
            "resources": [{"x": 0, "y": 0, "type": "shape", "key": "CbCbCbCb"}],
            "buildings": [],
        }
        grid = self.encoder.encode(payload)
        self.assertEqual(grid[self.channel["resource_shape_any"], 4, 4], 1.0)

    def test_out_of_window_entries_are_dropped(self):
        payload = {
            "resources": [{"x": 999, "y": 999, "type": "shape", "key": "CuCuCuCu"}],
            "buildings": [building("belt", -999, -999)],
        }
        self.assertEqual(self.encoder.encode(payload).sum(), 0.0)

    def test_multi_tile_building_covers_footprint(self):
        payload = {"resources": [], "buildings": [building("cutter", 0, 0, w=2, h=1)]}
        grid = self.encoder.encode(payload)
        cutter = self.channel["building_cutter"]
        self.assertEqual(grid[cutter, 4, 4], 1.0)
        self.assertEqual(grid[cutter, 4, 5], 1.0)
        self.assertEqual(grid[cutter].sum(), 2.0)

    def test_rotation_channel_is_set(self):
        payload = {"resources": [], "buildings": [building("belt", 0, 0, rotation=180)]}
        grid = self.encoder.encode(payload)
        self.assertEqual(grid[self.channel["rotation_180"], 4, 4], 1.0)
        self.assertEqual(grid[self.channel["rotation_0"], 4, 4], 0.0)

    def test_hub_and_unknown_buildings_are_separated(self):
        payload = {
            "resources": [],
            "buildings": [building("hub", 0, 0, w=2, h=2), building("wire", 3, 3)],
        }
        grid = self.encoder.encode(payload)
        self.assertEqual(grid[self.channel["building_hub"]].sum(), 4.0)
        self.assertEqual(grid[self.channel["building_other"]].sum(), 1.0)

    def test_occupancy_marks_full_footprint(self):
        payload = {"resources": [], "buildings": [building("hub", -2, -2, w=2, h=2)]}
        occupied = self.encoder.occupancy(payload)
        self.assertEqual(occupied.shape, (8, 8))
        self.assertEqual(occupied.sum(), 4)
        self.assertTrue(occupied[2, 2])
        self.assertFalse(occupied[0, 0])

    def test_resource_mask_filters_by_key(self):
        payload = {
            "resources": [
                {"x": 0, "y": 0, "type": "shape", "key": "CuCuCuCu"},
                {"x": 1, "y": 0, "type": "shape", "key": "RuRuRuRu"},
            ],
            "buildings": [],
        }
        mask = self.encoder.resource_mask(payload, kind="shape", key="CuCuCuCu")
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[4, 4])

    def test_compact_and_full_encode_identically(self):
        full = {
            "compact": False,
            "resources": [
                {"x": 0, "y": 0, "type": "shape", "key": "CuCuCuCu"},
                {"x": 2, "y": 1, "type": "color", "key": "red"},
            ],
            "buildings": [building("cutter", -1, 2, w=2, h=1, rotation=90)],
        }
        compact = {
            "compact": True,
            "resources": [[0, 0, "shape", "CuCuCuCu"], [2, 1, "color", "red"]],
            "buildings": [["cutter", -1, 2, 2, 1, 90]],
        }
        np.testing.assert_array_equal(
            self.encoder.encode(full), self.encoder.encode(compact)
        )

    def test_compact_occupancy_matches_full(self):
        full = {"compact": False, "resources": [], "buildings": [building("hub", -2, -2, 2, 2)]}
        compact = {"compact": True, "resources": [], "buildings": [["hub", -2, -2, 2, 2, 0]]}
        np.testing.assert_array_equal(
            self.encoder.occupancy(full), self.encoder.occupancy(compact)
        )

    def test_compact_resource_mask_filters_by_key(self):
        compact = {
            "compact": True,
            "resources": [[0, 0, "shape", "CuCuCuCu"], [1, 0, "shape", "RuRuRuRu"]],
            "buildings": [],
        }
        mask = self.encoder.resource_mask(compact, kind="shape", key="CuCuCuCu")
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[4, 4])

    def test_observation_stays_in_unit_range(self):
        payload = {
            "resources": [{"x": 0, "y": 0, "type": "shape", "key": "CuCuCuCu"}],
            "buildings": [building("belt", 0, 0)],
        }
        grid = self.encoder.encode(payload)
        self.assertTrue(np.all(grid >= 0.0) and np.all(grid <= 1.0))
        self.assertEqual(grid.dtype, np.float32)


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())
