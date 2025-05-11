import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tempfile
from summary_writer import SummaryWriter

def test_summary_writer_add_scalar_and_flush():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SummaryWriter(tmpdir, flush_secs=1, max_queue=1)
        writer.add_scalar('test_var', 123, 1)
        writer.flush()
        writer.close()
        # Check log file exists and contains correct header and entry
        files = [f for f in os.listdir(tmpdir) if f.startswith('log_') and f.endswith('.csv')]
        assert files, 'No log file created.'
        log_path = os.path.join(tmpdir, files[0])
        with open(log_path, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == 'time,var_name,scalar,step'
            assert any('test_var,123,1' in line for line in lines)
        # Simulate abrupt close and recovery
        writer2 = SummaryWriter(tmpdir, flush_secs=1, max_queue=1)
        writer2.add_scalar('recovery', 456, 2)
        writer2.flush()
        writer2.close()
        files2 = [f for f in os.listdir(tmpdir) if f.startswith('log_') and f.endswith('.csv')]
        assert files2, 'No log file created after recovery.'
        # Multiple writers in same dir
        writer3 = SummaryWriter(tmpdir, flush_secs=1, max_queue=1)
        writer4 = SummaryWriter(tmpdir, flush_secs=1, max_queue=1)
        writer3.add_scalar('multi', 789, 3)
        writer4.add_scalar('multi2', 101, 4)
        writer3.flush(); writer4.flush()
        writer3.close(); writer4.close()
        # Simulate log file corruption
        log_path = os.path.join(tmpdir, files[0])
        with open(log_path, 'wb') as f:
            f.write(b'corrupted')
        # Should not raise
        writer5 = SummaryWriter(tmpdir, flush_secs=1, max_queue=1)
        writer5.add_scalar('after_corrupt', 999, 5)
        writer5.flush()
        writer5.close()
