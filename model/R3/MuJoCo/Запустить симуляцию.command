#!/bin/zsh
cd -- "${0:A:h}" || exit 1
print 'R3 · Стенд Gruzdy / MuJoCo'
read 'load_mass?Масса манекена в кг [6]: '
read 'mode?1 — подъём на колёсах, 2 — езда и разворот [1]: '
opts=(--on-wheels)
[[ "$mode" == 2 ]] && opts+=(--drive --speedup 1)
python_gui=../../../VerticaliZaotr/.venv/bin/mjpython
if [[ ! -x "$python_gui" ]]; then
  python_gui=./.venv/bin/mjpython
fi
if [[ ! -x "$python_gui" ]]; then
  python_gui=$(command -v mjpython) || { print 'Установите зависимости по разделу «Перенос на другой компьютер» в Расчёт.md.'; exit 1; }
fi
exec "$python_gui" stand.py --viewer --mass "${load_mass:-6}" "${opts[@]}"
