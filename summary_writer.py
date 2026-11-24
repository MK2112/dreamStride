import time

class SummaryWriter:
    def __init__(self, log_dir, flush_secs, max_queue):
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self.max_queue = max_queue
        self._last_flush = time.time()
        stamp = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime())
        self.log_file = open(f'{self.log_dir}/log_{stamp}.csv', 'w')
        self.log_file.write('time,var_name,scalar,step\n')

    def add_scalar(self, name, scalar, step_):
        stamp = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime())
        self.log_file.write(f'{stamp},{name},{scalar},{step_}\n')
        if time.time() - self._last_flush >= self.flush_secs:
            self.flush()

    def flush(self):
        if self.log_file:
            self.log_file.flush()
        self._last_flush = time.time()

    def close(self):
        self.log_file.close()