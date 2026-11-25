"""Render actual MuJoCo states for the 8 kg run, alongside the three gravity curves."""
import math
from pathlib import Path

import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

from stand import Bench, camera, HERE

FONTS = Path("/System/Library/Fonts/Supplemental")
def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)

INK, SUB, PAPER = "#17353C", "#4C6B70", "#F6F7F3"
COLORS = {5: "#6082AE", 6: "#248D84", 8: "#B85D39"}
curves = {mass: Bench(mass).gravity_curve() for mass in COLORS}
bench = Bench(8, mobile=True)
frames = []
view = mj.MjvCamera()
mj.mjv_defaultCamera(view)
camera(view)
view.distance = 1.03
view.lookat[:] = (.14, 0, .32)

with mj.Renderer(bench.model, height=460, width=634) as renderer:
    def capture(b):
        if frames and b.d.time < len(frames) * 2.0 and b.d.time < 156.74:
            return
        b.sync_gears()
        renderer.update_scene(b.d, camera=view)
        shot = Image.fromarray(renderer.render())
        card = Image.new("RGB", (990, 614), PAPER)
        draw = ImageDraw.Draw(card)
        draw.text((23, 16), "R4   /   Подъём манекена", font=font(29, True), fill=INK)
        draw.text((25, 54), "412 × 252 × 436 мм сидя    ·    Gruzdy под сиденьем    ·    Четыре колеса Ø128", font=font(16), fill=SUB)
        draw.text((861, 28), "MuJoCo", font=font(22, True), fill=SUB)
        card.paste(shot, (18, 88))
        x = 679
        draw.text((x, 97), "Манекен 8 кг", font=font(25, True), fill=INK)
        angle = max(0., math.degrees(b.q("lift")))
        draw.text((x, 143), f"{angle:.1f}°".replace(".", ","), font=font(47, True), fill=INK)
        draw.text((x, 206), f"Ход каретки   {max(0., b.q('extension')*1000):.1f} мм".replace(".", ","), font=font(17), fill=SUB)
        draw.text((x, 234), f"Усилие привода   {abs(b.d.actuator_force[0]):.0f} Н", font=font(17), fill=SUB)
        draw.line((x,278,962,278), fill="#D0DBD7", width=1)
        draw.text((x, 297), "Усилие от веса, Н", font=font(19, True), fill=INK)
        left, top, right, bottom = 716, 348, 959, 491
        def xy(deg, force):
            return (left + deg / 75 * (right - left), bottom - force / 180 * (bottom - top))
        for force in (0, 80, 160):
            _, y = xy(0, force)
            draw.line((left, y, right, y), fill="#CFDAD7")
            draw.text((left-8, y), str(force), font=font(13), anchor="rm", fill=SUB)
        for mass, curve in curves.items():
            draw.line([xy(*p) for p in curve], fill=COLORS[mass], width=3)
        px, _ = xy(angle, 0)
        draw.line((px, top, px, bottom), fill="#778A8A")
        for deg in (0, 25, 50, 75):
            draw.text((xy(deg,0)[0], bottom+6), f"{deg}°", font=font(13), anchor="mt", fill=SUB)
        for index, mass in enumerate(curves):
            xx = left-20+index*91
            draw.line((xx,536,xx+20,536), fill=COLORS[mass], width=3)
            draw.text((xx+26, 536), f"{mass} кг", font=font(14), anchor="lm", fill=SUB)
        draw.text((23, 566), f"Время в модели   {b.d.time:04.1f} с".replace(".", ","), font=font(20, True), fill=INK)
        draw.text((x, 569), "Расчётный предел тяги   335 Н", font=font(17), fill=SUB)
        draw.rectangle((23, 599, 648, 601), fill="#CAD7D3")
        draw.rectangle((23, 599, 23+625*min(1,b.d.time/156.75), 601), fill="#22877C")
        frames.append(card)
    stats, _ = bench.trial(callback=capture)

frames[-1].save(HERE / "R4_8kg.png")
frames[0].save(HERE / "R4_8kg.gif", save_all=True, append_images=frames[1:],
               duration=[105]*(len(frames)-1)+[1200], loop=0, optimize=True, disposal=1)
print(len(frames), "frames", stats)
