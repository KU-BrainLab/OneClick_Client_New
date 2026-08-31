# -*- coding:utf-8 -*-
"""원격 제출 GUI 를 실행 파일 하나로 묶는다.

측정 PC 에는 이 실행 파일만 둔다 — 분석 파이프라인 소스는 넘기지 않는다.
파이썬도 필요 없다.

    pip install pyinstaller
    python build_remote_gui.py

결과: dist/OneClickRemote.exe (파일 이름은 마음대로 바꿔도 된다)

서버 주소는 remote_gui.py 의 SERVER 값이 실행 파일 안에 들어간다. 나중에
서버가 바뀌면 다시 빌드하거나, 측정 PC 의 ~/.oneclick_remote_gui.json 에
{"server": "..."} 를 넣어 덮으면 된다.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'OneClickRemote'


def main():
    try:
        import PyInstaller                                    # noqa: F401
    except ImportError:
        raise SystemExit('PyInstaller 가 없습니다: pip install pyinstaller')

    icon = os.path.join(HERE, 'remote_gui.ico')
    argv = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--windowed',                 # 콘솔 창 없이 GUI 만
        '--name', NAME,
        '--distpath', os.path.join(HERE, 'dist'),
        '--workpath', os.path.join(HERE, 'build'),
        '--specpath', os.path.join(HERE, 'build'),
        '--noconfirm',
    ]
    if os.path.exists(icon):
        # --icon 은 실행 파일 아이콘, --add-data 는 창 아이콘용 사본
        argv += ['--icon', icon, '--add-data', icon + os.pathsep + '.']
    for logo in ('logo_lab.png', 'logo_univ.png', 'logo_company.png'):
        lp = os.path.join(HERE, logo)
        if os.path.exists(lp):
            argv += ['--add-data', lp + os.pathsep + '.']
    ca = os.path.join(HERE, 'oneclick-ca.pem')
    if os.path.exists(ca):
        # 서버(자체 서명) 검증용 연구실 CA 공개 인증서 — https 필수 동반물
        argv += ['--add-data', ca + os.pathsep + '.']
    else:
        print('경고: oneclick-ca.pem 이 없습니다. 웹서버에서 인증서를 만들고 '
              'CA 공개 파일을 저장소 루트에 복사한 뒤 다시 빌드하세요 — '
              '없으면 https 접속이 인증서 오류로 실패합니다.')
    # 분석 파이프라인이 딸려 들어가지 않게 저장소를 탐색 경로에서 뺀다.
    # remote_gui.py 는 저장소 모듈을 import 하지 않으므로 이걸로 충분하다.
    argv += ['--paths', os.path.join(HERE, '_nothing')]
    argv += [os.path.join(HERE, 'remote_gui.py')]

    print('빌드 시작...')
    subprocess.check_call(argv, cwd=HERE)

    exe = os.path.join(HERE, 'dist', NAME + '.exe')
    print('\n완성: %s (%.1fMB)' % (exe, os.path.getsize(exe) / 1e6))
    shutil.rmtree(os.path.join(HERE, 'build'), ignore_errors=True)
    print('측정 PC 에는 이 파일 하나만 복사하면 됩니다.')


if __name__ == '__main__':
    main()
