@echo off
rem 바탕화면에 "OneClick 원격 제출" 바로가기를 만든다.
rem 콘솔 창 없이 뜨도록 pythonw 로 연결한다. 아무 PC 에서나 이 파일을
rem 더블클릭하면 된다 (파이썬에 requests 만 있으면 GUI 가 돈다).
setlocal
set "REPO=%~dp0"

rem pythonw 탐색: PATH -> 아나콘다 기본 경로 순서
set "PYW="
for /f "delims=" %%p in ('where pythonw 2^>nul') do if not defined PYW set "PYW=%%p"
if not defined PYW if exist "%LOCALAPPDATA%\anaconda3\pythonw.exe" set "PYW=%LOCALAPPDATA%\anaconda3\pythonw.exe"
if not defined PYW (
    echo pythonw 를 찾지 못했습니다. 파이썬 설치를 확인하세요.
    pause
    exit /b 1
)

powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\OneClick 원격 제출.lnk');" ^
  "$s.TargetPath = '%PYW%';" ^
  "$s.Arguments = '\"%REPO%remote_gui.py\"';" ^
  "$s.WorkingDirectory = '%REPO%';" ^
  "$s.IconLocation = '%REPO%remote_gui.ico';" ^
  "$s.Description = 'OneClick 측정 결과 원격 제출';" ^
  "$s.Save()"

if %errorlevel%==0 (echo 바탕화면에 바로가기를 만들었습니다.) else (echo 실패했습니다.)
pause
