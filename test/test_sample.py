#!/usr/bin/env python3
"""Sample test demonstrating benchmark tool usage."""

import unittest
import tempfile
import shutil
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dl_benchmark.datasets import DatasetLoader, SyntheticDataset
from dl_benchmark.rendering import RenderingEngine
from dl_benchmark.monitoring import ResourceMonitor


class TestSampleBenchmark(unittest.TestCase):
    """Sample test showing how to use the benchmark tool components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_synthetic_dataset_creation(self):
        """Test creating synthetic datasets."""
        # Test classification dataset
        dataset = SyntheticDataset(size=10, img_size=224, num_classes=1000)
        self.assertEqual(len(dataset), 10)
        
        # Test getting a sample
        img, label = dataset[0]
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 1000)
        
    def test_dataset_loader(self):
        """Test dataset loader functionality."""
        loader = DatasetLoader(test_samples=10)
        
        # Test synthetic classification dataset
        dataloader = loader.create_synthetic_classification_dataset(img_size=224)
        self.assertIsNotNone(dataloader)
        
        # Test getting a batch
        for batch_data, batch_labels in dataloader:
            self.assertEqual(batch_data.shape[1:], (3, 224, 224))
            break  # Just test first batch
            
    def test_rendering_engine(self):
        """Test rendering engine initialization."""
        engine = RenderingEngine()
        self.assertIsNotNone(engine)
        
        # Test that class lists are loaded
        self.assertGreater(len(engine.coco_classes), 0)
        self.assertGreater(len(engine.imagenet_classes), 0)
        
    def test_resource_monitor(self):
        """Test resource monitoring functionality."""
        monitor = ResourceMonitor(enable_gpu_monitoring=False)
        self.assertIsNotNone(monitor)
        
        # Test that monitor can be started and stopped
        thread = monitor.start_monitoring()
        self.assertIsNotNone(thread)
        
        # Let it run briefly
        import time
        time.sleep(0.1)
        
        monitor.stop_monitoring()
        
        # Test getting stats
        stats = monitor.get_resource_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('cpu', stats)
        self.assertIn('memory', stats)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_safe_time_value(self):
        """Test safe time value function."""
        from dl_benchmark.utils import safe_time_value
        
        # Test normal value
        self.assertEqual(safe_time_value(5.0), 5.0)
        
        # Test very small value
        self.assertEqual(safe_time_value(0.0), 0.001)
        
        # Test negative value
        self.assertEqual(safe_time_value(-1.0), 0.001)
        
    def test_calculate_fps(self):
        """Test FPS calculation."""
        from dl_benchmark.utils import calculate_fps
        
        # Test normal case
        fps = calculate_fps(10.0)  # 10ms should give 100 FPS
        self.assertEqual(fps, 100.0)
        
        # Test very fast case (should be capped)
        fps = calculate_fps(0.001)
        self.assertEqual(fps, 10000)  # Capped at max FPS


if __name__ == '__main__':
    # Run tests with more verbose output
    unittest.main(verbosity=2)