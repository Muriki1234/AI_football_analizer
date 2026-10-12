import ast
import unittest
import json
import tempfile
import types
import sys
from unittest.mock import patch
from pathlib import Path
import numpy as np
from server.pipeline.samurai_kinematics import project_target_positions, target_motion_summary


class SamuraiKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[100., 0., 750.], [0., 100., 100.], [0., 0., 1.]])
        self.boxes = {f: [19 + f * .1, 32, 2, 2] for f in range(30)}

    def test_motion_without_any_yolo_detections(self):
        points = project_target_positions(self.boxes, [self.H] * 30)
        source = Path(__file__).with_name('tasks.py')
        tree = ast.parse(source.read_text())
        names = {'_speed_fields', '_summary_for_range', '_compute_player_summary'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = {'__name__':'server.pipeline.tasks', '__package__':'server.pipeline', 'np': np, '_MATCH_SEGMENT_TYPES': {'match'}, '_find_matched_player': lambda *args: None}
        exec(compile(ast.Module(nodes, []), str(source), 'exec'), env)
        result = env['_compute_player_summary'](
            {'players': [{} for _ in range(30)]}, self.boxes, [0]*30,
            fps=10, segments=[{'type':'match','start_frame':0,'end_frame':30,'start_sec':0,'end_sec':3}],
            target_positions=points)
        self.assertAlmostEqual(result['total_distance_m'], 2.9)
        self.assertAlmostEqual(result['avg_speed_kmh'], 3.6)
        self.assertEqual(result['speed_samples'], 29)
        self.assertEqual(result['speed_reliability'], 'ok')
        self.assertEqual(result['by_segment'][0]['total_distance_m'], 2.9)

    def test_missing_calibration_is_not_zero_meters(self):
        points = project_target_positions(self.boxes, [None]*30)
        result = target_motion_summary(self.boxes, points, [(0,30)], 10)
        self.assertIsNone(result['total_distance_m'])
        self.assertIsNone(result['max_speed_kmh'])
        self.assertEqual(result['tracked_frames'],30)
        self.assertEqual(result['calibrated_target_frames'],0)

    def test_stationary_calibrated_player_is_valid_zero(self):
        points = {f:(20.,34.) for f in self.boxes}
        result = target_motion_summary(self.boxes, points, [(0,30)],10)
        self.assertEqual(result['total_distance_m'],0)
        self.assertEqual(result['avg_speed_kmh'],0)
        self.assertGreater(result['speed_samples'],0)

    def test_camera_pan_does_not_become_player_motion(self):
        boxes = {f:[19+f,32,2,2] for f in range(30)}
        matrices = [self.H.copy() for _ in boxes]
        for f,H in enumerate(matrices): H[0,2]-=f*100
        points=project_target_positions(boxes,matrices)
        result=target_motion_summary(boxes,points,[(0,30)],10)
        self.assertEqual(result['total_distance_m'],0)

    def test_match_ranges_never_bridge_skipped_frames(self):
        points={f:(20+f*.1,34) for f in self.boxes}
        result=target_motion_summary(self.boxes,points,[(0,10),(20,30)],10)
        self.assertAlmostEqual(result['total_distance_m'],1.8)

    def test_long_calibration_gap_is_not_extrapolated(self):
        points={0:(20,34),29:(30,34)}
        result=target_motion_summary(self.boxes,points,[(0,30)],10)
        self.assertIsNone(result['total_distance_m'])

    def test_invalid_projective_points_rejected(self):
        bad=self.H.copy();bad[2,:]=0
        self.assertEqual(project_target_positions(self.boxes,[bad]*30),{})
        boxes={0:[1000,1000,20,40]}
        self.assertEqual(project_target_positions(boxes,[self.H]),{})

    def test_coordinate_spike_does_not_add_fake_distance(self):
        points={f:(20+f*.1,34) for f in self.boxes}
        points[15]=(70,34)
        result=target_motion_summary(self.boxes,points,[(0,30)],10)
        self.assertLessEqual(result['total_distance_m'],2.9)
        self.assertLess(result['max_speed_kmh'],4)
        self.assertEqual(result['speed_rejected_count'],2)

    def test_current_calibration_and_actual_resolution_required(self):
        tree=ast.parse(Path(__file__).with_name('analysis_core.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ViewTransformer')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='add_transformed_position_to_tracks')
        class Transform:
            def __init__(self,H): self.m=H
        env={'np':np,'RobustPitchTransformer':Transform}
        exec(compile(ast.Module([method],[]),'projection-test','exec'),env)
        sizes=[]
        obj=types.SimpleNamespace(_last_transformer=None,homography_cache=None,
            SOCCANA_PITCH_COORDS={i:(i*100.,i*100.) for i in range(4)},
            compute_homography_ransac=lambda *args:self.H,
            is_homography_valid=lambda H,w,h:(sizes.append((w,h)) or True),
            smooth_homography=lambda H,w,h:H)
        env['add_transformed_position_to_tracks'](obj,{'players':[{},{}]},
            [{i:(i*10.,i*10.) for i in range(4)},{}],image_size=(960,544))
        self.assertTrue(all(size==(960.,544.) for size in sizes))
        self.assertIsNotNone(obj.frame_homographies[0])
        self.assertIsNone(obj.frame_homographies[1])

    def test_heatmap_and_minimap_do_not_need_yolo_match(self):
        tree=ast.parse(Path(__file__).with_name('tasks.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_export_position_jsons')
        fn.returns=None
        for arg in fn.args.args: arg.annotation=None
        env={'__name__':'server.pipeline.tasks','__package__':'server.pipeline',
             '_find_matched_player':lambda *args:None}
        exec(compile(ast.Module([fn],[]),'export-test','exec'),env)
        storage=types.ModuleType('server.storage.r2')
        storage.upload_to_r2=lambda *args:None
        with patch.dict(sys.modules, {'server.storage.r2':storage}), tempfile.TemporaryDirectory() as d:
            points=project_target_positions(self.boxes,[self.H]*30)
            env['_export_position_jsons']('test',{'players':[{}]*30,'ball':[{}]*30},
                self.boxes,[0]*30,{},10,30,Path(d),target_positions=points)
            minimap=json.loads((Path(d)/'minimap_positions.json').read_text())
            heatmap=json.loads((Path(d)/'heatmap_positions.json').read_text())
            self.assertGreater(len(heatmap['positions']),0)
            self.assertTrue(all(any(p.get('tr') for p in frame) for frame in minimap['frames']))


if __name__=='__main__': unittest.main()
