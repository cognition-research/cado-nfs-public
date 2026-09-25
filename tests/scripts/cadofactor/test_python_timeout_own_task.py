#!/usr/bin/env python3
"""
Unit test for the timeout check of ClientServerTask: the task times out,
counts, and resubmits only its own workunits, not the assigned workunits
of another task that share the workunits table.
"""

import json
import logging
import unittest

from cadofactor import cadologger  # noqa: F401
from cadofactor.cadotask import Polysel1Task


class FakeWuAccess:
    def __init__(self, wuids):
        self.rows = [{"wuid": w, "wu": json.dumps({"id": w})} for w in wuids]
        self.created = []
        self.cancelled = []

    def query(self, **conditions):
        if "lt" in conditions:
            return list(self.rows)
        return []

    def create(self, wu, commit=True):
        self.created.append(wu.get_id())

    def cancel(self, wuid, commit=True):
        self.cancelled.append(wuid)


class DictWithCommit(dict):
    def update(self, other, commit=True):
        dict.update(self, other)


def make_task(wuar):
    task = Polysel1Task.__new__(Polysel1Task)
    task.params = {"name": "run", "wutimeout": 1, "wutimeoutcheck": 0,
                   "maxtimedout": 100, "maxresubmit": 5}
    task.state = DictWithCommit(wu_submitted=2, wu_received=0,
                                wu_timedout=0, start_real_time=1.0)
    task.logger = logging.getLogger("test")
    task.wuar = wuar
    task.last_timeout_check = 0
    return task


class TestTimeoutOwnTask(unittest.TestCase):
    def test_other_task_workunit_is_left_alone(self):
        wuar = FakeWuAccess(["run_polyselect1_0-120", "run_polyselect2_0-1",
                             "other_polyselect1_120-240__R2"])
        task = make_task(wuar)
        task.resubmit_timed_out_wus()
        self.assertEqual(wuar.cancelled, ["run_polyselect1_0-120",
                                          "other_polyselect1_120-240__R2"])
        self.assertEqual(wuar.created, ["run_polyselect1_0-120__R2",
                                        "run_polyselect1_120-240__R3"])
        self.assertEqual(task.state["wu_timedout"], 2)
        self.assertEqual(task.state["wu_submitted"], 4)
        self.assertEqual(task.get_number_outstanding_wus(), 2)


if __name__ == '__main__':
    unittest.main()
