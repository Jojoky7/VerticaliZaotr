@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul || (echo Нужен Python 3.10+ : https://www.python.org/downloads/  ^(при установке отметь "Add python.exe to PATH"^) & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  echo Первый запуск: создаю окружение Python...
  %PY% -m venv .venv || (echo Не удалось создать окружение & pause & exit /b 1)
)
".venv\Scripts\python.exe" -c "import mujoco, numpy, PIL" >nul 2>nul || (
  echo Ставлю библиотеки, 1-2 минуты, нужен интернет...
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (echo Не удалось установить библиотеки & pause & exit /b 1)
)
echo Симулятор запущен. Чтобы остановить - закрой это окно.
".venv\Scripts\python.exe" simulator.py
pause
