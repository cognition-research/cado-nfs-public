#!/usr/bin/env python3
"""
Unit test for the upload backlog of cado-nfs-client.py: a 200 (fresh or
already received) and a 409 (server holds a different result) both
finish the upload and clean up, any other status is retried and kept in
the backlog.
"""

import importlib.util
import logging
import os
import sys
import unittest

CLIENT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "cado-nfs-client.py")
sys.argv[0] = CLIENT
spec = importlib.util.spec_from_file_location("cado_nfs_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

SETTINGS = {"POSTRESULTPATH": "/upload",
            "DOWNLOADRETRY": "0.01",
            "DOWNLOADRETRYMIN": None}


class FakeResponse(object):
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text
        self.content = text.encode()

    def __bool__(self):
        # like requests.Response: falsy for any error status
        return self.status_code < 400


class FakePeer(object):
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = 0

    def get_index(self):
        return 0

    def get_url(self, ep=None):
        return "http://fake/upload"

    def post(self, ep, **kwargs):
        self.posts += 1
        return self.responses.pop(0)


class FakeWorkunit(object):
    def __init__(self, peer):
        self.peer = peer

    def get_id(self):
        return "wu1"

    def get_peer(self):
        return self.peer

    def is_stale(self):
        return False


class FakeProcessor(object):
    def __init__(self, peer):
        self.workunit = FakeWorkunit(peer)
        self.cleaned_up = False

    def build_answer(self):
        return {}

    def cleanup(self):
        self.cleaned_up = True


class FakeServerPool(object):
    servers = [None]


class TestClientUpload(unittest.TestCase):
    def run_uploader(self, responses):
        peer = FakePeer(responses)
        processor = FakeProcessor(peer)
        uploader = client.ResultUploader(SETTINGS, FakeServerPool())
        uploader.schedule_upload(processor)
        with self.assertLogs(level=logging.INFO) as logs:
            uploader.process_uploads()
        return uploader, processor, peer, logs

    def test_fresh_upload(self):
        responses = [FakeResponse(200, '{"message": "upload completed"}')]
        uploader, processor, peer, logs = self.run_uploader(responses)
        self.assertEqual(peer.posts, 1)
        self.assertTrue(processor.cleaned_up)
        self.assertEqual(uploader.backlog_size, 0)
        self.assertEqual(uploader.upload_backlog, [[]])

    def test_already_received(self):
        responses = [FakeResponse(200,
                                  '{"message": "upload already received"}')]
        uploader, processor, peer, logs = self.run_uploader(responses)
        self.assertEqual(peer.posts, 1)
        self.assertTrue(processor.cleaned_up)
        self.assertEqual(uploader.backlog_size, 0)
        self.assertEqual(uploader.upload_backlog, [[]])
        self.assertIn("INFO:root:Upload of wu1 succeeded:"
                      ' {"message": "upload already received"}',
                      logs.output)

    def test_conflict(self):
        responses = [FakeResponse(409, '{"code": 409}')]
        uploader, processor, peer, logs = self.run_uploader(responses)
        self.assertEqual(peer.posts, 1)
        self.assertTrue(processor.cleaned_up)
        self.assertEqual(uploader.backlog_size, 0)
        self.assertEqual(uploader.upload_backlog, [[]])
        self.assertTrue(any(m.startswith("ERROR:root:Server refused the"
                                         " result of wu1")
                            for m in logs.output))

    def test_other_status_is_retried(self):
        # what an old server answers to a duplicate upload
        responses = [FakeResponse(403, '{"code": 403}')] * 20
        uploader, processor, peer, logs = self.run_uploader(responses)
        self.assertGreater(peer.posts, 1)
        self.assertFalse(processor.cleaned_up)
        self.assertEqual(uploader.backlog_size, 1)
        self.assertEqual(uploader.upload_backlog, [[processor]])


if __name__ == "__main__":
    unittest.main()
