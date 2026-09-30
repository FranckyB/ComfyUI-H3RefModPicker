"""Run directly with ComfyUI's Python; uses only generated temporary media."""
import builtins
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import imageio.v2 as imageio
import numpy as np
import torch

source = Path(__file__).resolve().parents[1] / 'py/refmod_common.py'
spec = importlib.util.spec_from_file_location('refmod_video_under_test', source)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class VideoLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='refmod-loader-', dir=os.environ.get('REFMOD_TEST_TMPDIR'))
        cls.root = Path(cls.tmp.name)
        cls.paths = []
        for codec, pixel_format in [('libx264', 'yuv420p'), ('libx265', 'yuv420p10le')]:
            path = cls.root / f'{codec}.mov'
            cls.write_video(path, codec, pixel_format)
            cls.paths.append(path)

    @staticmethod
    def write_video(path, codec, pixel_format, rotation=0):
        with av.open(str(path), 'w', format='mov') as container:
            stream = container.add_stream(codec, rate=6)
            stream.width = 160
            stream.height = 96
            stream.pix_fmt = pixel_format
            if rotation:
                stream.set_display_rotation(rotation)
            stream.options = {'preset': 'ultrafast'}
            if codec == 'libx265':
                stream.options['x265-params'] = 'log-level=error:pools=1:frame-threads=1'
            for index in range(6):
                rgb = np.full((96, 160, 3), (index * 30, 100, 170), dtype=np.uint8)
                rgb[:24, :40] = (240, 30, 20)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pyav_handles_h264_and_10bit_hevc_without_other_loaders(self):
        for path in self.paths:
            with self.subTest(path=path.name), patch.object(imageio, 'get_reader', side_effect=AssertionError('Unexpected fallback')):
                output = loader.load_video_file(str(path), 2, 80)
                self.assertEqual(output.shape, (2, 48, 80, 3))
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(0 <= output.min() <= output.max() <= 1)
                self.assertGreater(float(output[-1, :, :, 0].mean()), float(output[0, :, :, 0].mean()) + 0.3)

    def test_imageio_fallback_handles_both_codecs(self):
        original = builtins.__import__
        def without_pyav(name, *args, **kwargs):
            if name == 'av':
                raise ImportError('PyAV intentionally unavailable for fallback test')
            if name == 'cv2':
                raise AssertionError('OpenCV should not be imported')
            return original(name, *args, **kwargs)
        for path in self.paths:
            with self.subTest(path=path.name), patch('builtins.__import__', side_effect=without_pyav):
                output = loader.load_video_file(str(path), 2, 80)
                self.assertEqual(output.shape, (2, 48, 80, 3))
                self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(hasattr(av.video.stream.VideoStream, 'set_display_rotation'),
                         'Writing rotation metadata for this fixture requires PyAV 18.1+')
    def test_phone_rotation_metadata_is_applied(self):
        rotated = self.root / 'rotated.mov'
        self.write_video(rotated, 'libx264', 'yuv420p', rotation=90)
        plain = loader.load_video_file(str(self.paths[0]), 2, 80)
        actual = loader.load_video_file(str(rotated), 2, 80)
        self.assertEqual(actual.shape, (2, 80, 48, 3))
        torch.testing.assert_close(actual, torch.rot90(plain, 1, (1, 2)), atol=0, rtol=0)

    def test_known_unknown_and_inaccurate_frame_counts_stay_bounded(self):
        def stream():
            for index in range(100):
                yield np.full((2, 3, 3), index, dtype=np.uint8)
        for count in (100, 0, 2, math.inf, math.nan):
            with self.subTest(count=count):
                result = loader._sample_video_frames(stream(), count, 4, lambda frame: frame)
                again = loader._sample_video_frames(stream(), count, 4, lambda frame: frame)
                self.assertEqual(result.shape, (4, 2, 3, 3))
                self.assertTrue(torch.equal(result, again))
                self.assertTrue((result[1:, 0, 0, 0] > result[:-1, 0, 0, 0]).all())
                if count == 100:
                    self.assertEqual((result[:, 0, 0, 0] * 255).round().tolist(), [0, 33, 66, 99])

    def test_both_error_causes_are_reported_and_decoders_close(self):
        closed = []
        class Container:
            streams = type('Streams', (), {'video': []})()
            def __enter__(self): return self
            def __exit__(self, *args): closed.append('pyav')
        class Reader:
            def get_meta_data(self): return {'nframes': 0}
            def __iter__(self): return iter(())
            def close(self): closed.append('imageio')
        with patch.object(av, 'open', return_value=Container()), patch.object(imageio, 'get_reader', return_value=Reader()):
            with self.assertRaisesRegex(RuntimeError, 'PyAV:.*no video stream.*ImageIO/FFmpeg:.*no decodable frames'):
                loader.load_video_file('unused-filename.mov', 2, 80)
        self.assertEqual(closed, ['pyav', 'imageio'])

    def test_missing_backends_report_import_errors(self):
        original = builtins.__import__

        def without_backends(name, *args, **kwargs):
            if name in ('av', 'imageio.v2'):
                raise ModuleNotFoundError(f'No module named {name!r}')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=without_backends):
            with self.assertRaisesRegex(RuntimeError, 'PyAV: No module.*ImageIO/FFmpeg: No module'):
                loader.load_video_file('unused-filename.mov', 2, 80)

    def test_memory_failure_is_not_retried_with_another_backend(self):
        with patch.object(av, 'open', side_effect=MemoryError('allocation failed')):
            with patch.object(imageio, 'get_reader', side_effect=AssertionError('Unexpected fallback')):
                with self.assertRaises(MemoryError):
                    loader.load_video_file('unused-filename.mov', 2, 80)

    def test_invalid_bounds_fail_before_opening_any_file(self):
        with patch.object(av, 'open', side_effect=AssertionError('Must not open a file')):
            for kwargs in ({'max_frames': 0}, {'max_edge': 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    loader.load_video_file('unused-filename.mov', **kwargs)


if __name__ == '__main__':
    unittest.main()
