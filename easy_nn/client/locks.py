"""
Lock-file control, kept exactly as it works in any_nn -- except the files live
next to your script, not on the executor.

Drop ``pause.lock`` in the output directory and training pauses; remove it and
it resumes.  ``do_eval.lock`` and ``save_checkpoint.lock`` fire once and are
deleted, same as before.
"""

from __future__ import annotations

import os
import threading

PAUSE = "pause.lock"
EVAL = "do_eval.lock"
SAVE = "save_checkpoint.lock"


class LockWatcher(threading.Thread):
    def __init__(self, output_dir: str, send_control, interval=0.5):
        super().__init__(name="easy-nn-locks", daemon=True)
        self.output_dir = output_dir
        self.send_control = send_control
        self.interval = interval
        self._stop = threading.Event()
        self._paused = False

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self._poll()
            except OSError:
                pass

    def _poll(self):
        pause_present = os.path.isfile(os.path.join(self.output_dir, PAUSE))
        if pause_present and not self._paused:
            self._paused = True
            self.send_control("pause")
        elif not pause_present and self._paused:
            self._paused = False
            self.send_control("resume")

        for filename, action in ((EVAL, "eval"), (SAVE, "save")):
            path = os.path.join(self.output_dir, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    continue
                self.send_control(action)

    def stop(self):
        self._stop.set()
