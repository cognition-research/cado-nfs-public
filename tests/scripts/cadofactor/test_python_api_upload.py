#!/usr/bin/env python3
"""
Unit test for the /upload endpoint of the api server: a fresh upload is
stored, an identical re-upload is acknowledged as already received, and
an upload that clashes with a different existing file is refused with
409.
"""

import io
import json
import os
import tempfile
import unittest

from cadofactor import cadologger  # noqa: F401
from cadofactor import wudb
from cadofactor.api_server import ApiServer
from cadofactor.workunit import Workunit, WuStatus


class TestApiUpload(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        dbpath = os.path.join(self.tmpdir.name, "wu.db")
        db = wudb.DBFactory(f"db:sqlite3://{dbpath}", create=True)
        self.wuaccess = wudb.WuAccess(db)
        self.wuaccess.create_tables()
        self.wuaccess.create(Workunit(id="wu1"))
        self.assertIsNotNone(self.wuaccess.assign("client1"))
        self.uploaddir = os.path.join(self.tmpdir.name, "upload")
        self.server = ApiServer("localhost", 0, db,
                                threaded=False,
                                uploaddir=self.uploaddir,
                                nrsubdir=0,
                                whitelist=["127.0.0.1"])
        self.client = self.server.test_client()

    def tearDown(self):
        self.server.server.server_close()
        self.tmpdir.cleanup()

    def upload(self, content):
        fileinfo = {"wu1.out": {"WUid": "wu1", "key": "RESULT0"}}
        data = {"clientid": "client1",
                "WUid": "wu1",
                "fileinfo": json.dumps(fileinfo),
                "wu1.out": (io.BytesIO(content), "wu1.out")}
        return self.client.post("/upload", data=data)

    def wu_status(self):
        return self.wuaccess.query(eq={"wuid": "wu1"})[0]["status"]

    def stored_content(self):
        with open(os.path.join(self.uploaddir, "wu1.out"), "rb") as f:
            return f.read()

    def test_new_upload(self):
        resp = self.upload(b"result data\n")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"message": "upload completed"})
        self.assertEqual(self.stored_content(), b"result data\n")
        self.assertEqual(self.wu_status(), WuStatus.RECEIVED_OK)

    def test_identical_duplicate(self):
        self.upload(b"result data\n")
        resp = self.upload(b"result data\n")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(),
                         {"message": "upload already received"})
        self.assertEqual(self.stored_content(), b"result data\n")
        self.assertEqual(self.wu_status(), WuStatus.RECEIVED_OK)

    def test_identical_duplicate_records_result(self):
        # file stored by an earlier upload that never reached result()
        with open(os.path.join(self.uploaddir, "wu1.out"), "wb") as f:
            f.write(b"result data\n")
        resp = self.upload(b"result data\n")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(),
                         {"message": "upload already received"})
        self.assertEqual(self.wu_status(), WuStatus.RECEIVED_OK)

    def test_differing_duplicate(self):
        self.upload(b"result data\n")
        resp = self.upload(b"other data\n")
        self.assertEqual(resp.status_code, 409)
        body = resp.get_json()
        self.assertEqual(body["code"], 409)
        self.assertEqual(body["description"],
                         "File already exists with different content")
        self.assertEqual(self.stored_content(), b"result data\n")


if __name__ == "__main__":
    unittest.main()
