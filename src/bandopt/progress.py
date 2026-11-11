"""A one-line terminal progress bar for long exhaustive searches."""

import sys
import time

from bandopt.timing import format_duration


class ProgressBar:
    """Redraw a single status line on stderr, at most every `interval` seconds.

    The bar counts subsets. `done` starts at the work already finished (for
    example chunks completed in an earlier session), so a resumed search shows
    its true percentage. The rate and time left use only this session's work.

    It draws only when stderr is a terminal, so output redirected to a log file
    stays clean. Call `message()` instead of print() while the bar is shown.
    `sync` is called before each redraw; pass torch.cuda.synchronize on a GPU so
    the bar shows finished work rather than work still queued on the device.
    """

    def __init__(self, total, done=0, label="", interval=0.5, width=30,
                 stream=None, enabled=None, sync=None):
        self.total = max(total, 1)
        self.done = done
        self.label = label
        self.interval = interval
        self.width = width
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.sync = sync
        self.chunk_text = ""
        self.session_done = 0
        self.began = time.perf_counter()
        self.last_draw = 0.0
        self.line_length = 0

    def set_chunk(self, text):
        self.chunk_text = text

    def skip(self, count):
        """Count work that was already done earlier, without changing the rate."""
        self.done += count

    def update(self, count):
        self.done += count
        self.session_done += count
        now = time.perf_counter()
        if now - self.last_draw >= self.interval:
            self.draw(now)

    def render(self, now=None):
        elapsed = (now or time.perf_counter()) - self.began
        fraction = min(self.done / self.total, 1.0)
        filled = int(fraction * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.session_done / elapsed if elapsed > 0 else 0.0
        left = (self.total - self.done) / rate if rate > 0 else float("nan")
        parts = [
            self.label,
            f"[{bar}] {100 * fraction:5.1f}%",
            f"{self.done:,}/{self.total:,}",
            f"{rate:,.0f}/s" if rate else "-/s",
            f"elapsed {format_duration(elapsed)}",
            f"left {format_duration(left)}",
            self.chunk_text,
        ]
        return "  ".join(part for part in parts if part)

    def draw(self, now=None):
        if not self.enabled:
            return
        if self.sync:
            self.sync()
        now = now or time.perf_counter()
        line = self.render(now)
        padding = " " * max(self.line_length - len(line), 0)
        self.stream.write("\r" + line + padding)
        self.stream.flush()
        self.line_length = len(line)
        self.last_draw = now

    def clear(self):
        if self.enabled and self.line_length:
            self.stream.write("\r" + " " * self.line_length + "\r")
            self.stream.flush()
            self.line_length = 0

    def message(self, text):
        """Print a line above the bar, then redraw the bar."""
        self.clear()
        print(text, flush=True)
        self.draw()

    def close(self):
        self.draw()
        if self.enabled and self.line_length:
            self.stream.write("\n")
            self.stream.flush()
            self.line_length = 0
