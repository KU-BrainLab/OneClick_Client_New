# -*- coding: utf-8 -*-
"""LogFollower(remote_gui) / _LogFollower(remote_submit) 단위 테스트 — 서버 없이.

서버는 run.log 의 마지막 8000바이트만 돌려주므로, 글자 수만 비교하던 옛 방식은 로그가
창보다 커지는 순간 멈췄다. 두 구현이 같은 규칙으로 바이트 단위로 따라가는지 본다.

    python -m unittest tests.test_log_follower -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WINDOW = 8000


def status_for(log, window=WINDOW):
    """analysis_server 가 돌려주는 모양으로."""
    b = log.encode('utf-8')
    frm = max(0, len(b) - window)
    return {'status': 'running', 'log_tail': b[frm:].decode('utf-8', 'replace'),
            'log_bytes': len(b), 'log_from': frm}


def followers():
    import remote_gui
    import remote_submit
    return [remote_gui.LogFollower, remote_submit._LogFollower]


class LogFollowerTests(unittest.TestCase):
    def test_follows_growth_beyond_window_without_gaps_or_duplicates(self):
        for cls in followers():
            f = cls()
            log, out, skipped = '', [], 0
            for i in range(3000):                       # 10바이트 × 3000 = 30000바이트 > 창
                log += 'line %04d\n' % i
                if i % 37 == 0:                         # 조회 사이 370바이트 < 창
                    new, sk = f.take(status_for(log))
                    out.append(new)
                    skipped += sk
            new, sk = f.take(status_for(log))
            out.append(new)
            skipped += sk
            self.assertEqual(''.join(out), log, cls)
            self.assertEqual(skipped, 0, cls)
            self.assertEqual(f.take(status_for(log)), ('', 0), cls)      # 더 안 늘면 아무것도 없다

    def test_reports_skipped_bytes_when_growth_exceeds_window(self):
        for cls in followers():
            f = cls()
            first = 'a' * 100 + '\n'
            self.assertEqual(f.take(status_for(first)), (first, 0))
            log = first + 'b' * 20000 + '\n'
            new, sk = f.take(status_for(log))
            self.assertEqual(sk, len(log) - len(first) - WINDOW, cls)
            self.assertEqual(len(new.encode('utf-8')), WINDOW, cls)
            self.assertTrue(new.endswith('b\n'), cls)
            self.assertEqual(f.take(status_for(log)), ('', 0), cls)

    def test_multibyte_text_is_cut_on_byte_boundary(self):
        for cls in followers():
            f = cls()
            line = '한글 로그 줄 — 방추파 12개 검출\n'          # 글자당 1~3바이트 섞임
            log, out = '', []
            for i in range(600):
                log += line
                if i % 5 == 4:                          # 조회 사이 5줄(약 230바이트)
                    out.append(f.take(status_for(log))[0])
            self.assertEqual(''.join(out), log, cls)

    def test_old_server_without_log_bytes_uses_tail_length(self):
        for cls in followers():
            f = cls()
            self.assertEqual(f.take({'log_tail': 'ab'}), ('ab', 0), cls)
            self.assertEqual(f.take({'log_tail': 'abcd'}), ('cd', 0), cls)
            self.assertEqual(f.take({'log_tail': 'abcd'}), ('', 0), cls)
            self.assertEqual(f.take({'log_tail': 'x'}), ('', 0), cls)       # 창이 밀림
            self.assertEqual(f.take({'log_tail': 'xy'}), ('y', 0), cls)

    def test_missing_fields_are_tolerated(self):
        for cls in followers():
            f = cls()
            self.assertEqual(f.take({}), ('', 0), cls)
            self.assertEqual(f.take({'status': 'queued', 'log_tail': '', 'log_bytes': 0, 'log_from': 0}),
                             ('', 0), cls)
            self.assertEqual(f.take({'log_tail': 'abc', 'log_bytes': 3}), ('abc', 0), cls)   # log_from 없음


if __name__ == '__main__':
    unittest.main()
