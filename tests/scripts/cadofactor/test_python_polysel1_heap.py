#!/usr/bin/env python3
"""
Unit test for the best-polynomials heap of Polysel1Task: a polynomial
that is already in the heap is not added a second time, and a DB that
holds a polynomial twice is compacted when the heap is loaded from it.
"""

import logging
import unittest

from cadofactor import cadologger  # noqa: F401
from cadofactor.cadotask import Polynomials, Polysel1Task


def make_poly(c6, exp_E):
    return Polynomials([
        "n: 1000000000000000000000000000000000000000000000000000000000000",
        "c6: %d" % c6, "c5: 1", "c4: 1", "c3: 1", "c2: 1", "c1: 1", "c0: 1",
        "Y1: 1", "Y0: -7",
        "# exp_E %f" % exp_E,
    ])


class DictWithCommit(dict):
    """ The part of DictDbCachedAccess that the heap uses """

    def clear(self, args=None, commit=True):
        dict.clear(self)

    def update(self, other, commit=True):
        dict.update(self, other)


def make_task(nrkeep, db):
    task = Polysel1Task.__new__(Polysel1Task)
    task.params = {"nrkeep": nrkeep}
    task.logger = logging.getLogger("test")
    task.poly_heap = []
    task.poly_heap_ids = set()
    task.best_polynomials = db
    return task


class TestPolysel1Heap(unittest.TestCase):
    def test_duplicate_is_not_added(self):
        task = make_task(3, DictWithCommit())
        self.assertEqual(task._add_poly_heap(make_poly(1, 60.0)), "0")
        self.assertIsNone(task._add_poly_heap(make_poly(1, 60.0)))
        self.assertEqual(task._add_poly_heap(make_poly(2, 61.0)), "1")
        self.assertEqual(len(task.poly_heap), 2)

    def test_evicted_poly_can_return(self):
        task = make_task(2, DictWithCommit())
        task._add_poly_heap(make_poly(1, 60.0))
        task._add_poly_heap(make_poly(2, 62.0))
        self.assertEqual(task._add_poly_heap(make_poly(3, 61.0)), "1")
        self.assertIsNone(task._add_poly_heap(make_poly(2, 62.0)))
        # c6=3 is evicted here and may enter again afterwards
        self.assertEqual(task._add_poly_heap(make_poly(4, 59.0)), "1")
        self.assertEqual(task._add_poly_heap(make_poly(3, 59.5)), "0")

    def test_db_with_duplicates_is_compacted(self):
        polys = [make_poly(1, 60.0), make_poly(1, 60.0), make_poly(2, 61.0)]
        db = DictWithCommit((str(i), str(p)) for i, p in enumerate(polys))
        task = make_task(10, db)
        task.import_existing_polynomials()
        task._check_best_polynomials()
        task._compare_heap_db()
        self.assertEqual(sorted(db), ["0", "1"])
        self.assertEqual(
            sorted(p.polyf[6] for p in task.get_raw_polynomials()), [1, 2])

    def test_clean_db_is_left_alone(self):
        polys = [make_poly(1, 60.0), make_poly(2, 61.0)]
        db = DictWithCommit((str(i), str(p)) for i, p in enumerate(polys))
        before = dict(db)
        task = make_task(10, db)
        task.import_existing_polynomials()
        task._compare_heap_db()
        self.assertEqual(dict(db), before)


if __name__ == "__main__":
    unittest.main()
