import unittest

loader = unittest.TestLoader()
suite = loader.discover(start_dir="./src/tests", pattern="test_*.py")

runner = unittest.TextTestRunner()
runner.run(suite)
