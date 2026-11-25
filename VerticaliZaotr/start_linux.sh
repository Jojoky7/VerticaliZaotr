#!/bin/bash
# Симулятор V6: запуск: bash start_linux.sh. Откроется страница в браузере (видео + пульт).
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен Python 3.10+ — скачай с https://www.python.org/downloads/ и запусти этот файл ещё раз."; read -r; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение Python..."
  python3 -m venv .venv || { echo "Не удалось создать окружение."; read -r; exit 1; }
fi
if ! .venv/bin/python -c "import mujoco, numpy, PIL" 2>/dev/null; then
  echo "Ставлю библиотеки (1-2 минуты, нужен интернет)..."
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "Не удалось установить библиотеки."; read -r; exit 1; }
fi
echo "Симулятор запущен. Чтобы остановить — закрой это окно."
exec .venv/bin/python simulator.py
