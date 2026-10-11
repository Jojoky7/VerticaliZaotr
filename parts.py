"""Inspection of physical assemblies, with geometry and mass from the loaded model.

Materials are design assumptions. Body inertials describe assemblies, so we do
not invent a separate mass for every decorative surface or infer material from colour.
"""
import mujoco
import numpy as np


class Parts:
    def __init__(self, m, frame_mat, actuator):
        self.m = m
        self.groups, self.geom_part = {}, {}
        self.frame = m.body("frame").id
        metal = "Сталь" if frame_mat == "steel" else "Алюминий 6061-T6"
        def add(key, label, bodies, material, note="Масса узла из инерциальной модели; материалы приняты при проектировании.", geoms=None):
            bodies = list(bodies)
            geoms = list(geoms) if geoms is not None else [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_group[g] in (1, 2)]
            if not geoms:
                return
            self.groups[key] = dict(id=key, name=label, material=material, note=note,
                                    mass_kg=float(m.body_mass[bodies].sum()) if bodies else None,
                                    geoms=geoms, bodies=bodies)
            for g in geoms:
                self.geom_part[g] = key
        names = {b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)}
        add("frame", "Рама и закреплённые узлы", [self.frame], metal + "; батарея, электроника, накладки",
            "Масса всего неподвижного узла рамы, включая закреплённые батарею и электронику. Отдельные массы поверхностей не заданы.")
        for name, label, material in (("seat", "Сиденье", metal + ", фанера, пеноматериал, обивка"),
                                       ("back", "Спинка и подлокотники", metal + ", пеноматериал, обивка"),
                                       ("lower_link", "Нижние тяги", "Сталь")):
            add(name, label, [m.body(name).id], material)
        for wheel, label in (("drive_left", "Заднее левое колесо"), ("drive_right", "Заднее правое колесо"),
                              ("front_left", "Переднее левое колесо"), ("front_right", "Переднее правое колесо")):
            add(wheel, label, [m.body(wheel).id], "Резиновая шина, алюминиевый обод, стальная ступица")
            side = wheel.removeprefix("front_")
            if wheel.startswith("front_"):
                fork = m.body("front_fork_" + side).id
                add("front_fork_" + side, "Вилка переднего колеса: " + ("левая" if side == "left" else "правая"), [fork], metal)
        for suffix, label in (("", "Актуатор левый" if actuator["count"] == 2 else "Актуатор"), ("_right", "Актуатор правый")):
            if suffix and actuator["count"] != 2:
                continue
            bodies = [m.body("actuator_cylinder" + suffix).id, m.body("actuator_rod" + suffix).id]
            add("actuator" + suffix, label, bodies, "Металлический корпус, стальной шток; точные марки не заданы",
                f"Проектный привод: {actuator['unit_force']:.0f} Н на выдвижение, {actuator['unit_retract_force']:.0f} Н на втягивание; ход 300 мм. Размеры и масса требуют сверки с выбранным изделием.")
        human = [b for b, n in names.items() if n.startswith("human")]
        add("human", "Человек", human, "Условные жёсткие сегменты тела", "Масса задана пользователем; распределение по сегментам — антропометрическая модель.")
        # CAD components with their own inertials, when that construction is selected.
        known = {b for part in self.groups.values() for b in part["bodies"]}
        for b, name in names.items():
            if b and b not in known:
                label, material, note = name, "Материал узла не задан", "Масса узла из инерциальной модели."
                if name.startswith("Motor150_"):
                    label = "Мотор и редуктор: " + ("левый" if name.endswith("Left") else "правый")
                    material = "Металлический корпус, обмотки, стальные шестерни; точные марки не заданы"
                elif name == "Battery_24V_20Ah":
                    label, material = "Батарея 25,6 В · 20 А·ч", "LiFePO₄; корпус и защитный чехол"
                    note = "Модельная масса батареи с упаковкой; запас энергии 512 Вт·ч."
                add(name, label, [b], material, note)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
                label = "Порог 30 мм" if name == "threshold_30mm" else "Пол" if name == "floor" else name
                add("track_" + name, label, [], "Материал не задан", "Неподвижный объект испытательной трассы; масса не моделируется.", [g])
        # Whole-chair overview: exclude all human segments and the test track.
        chair_bodies = [b for b in names if b and b not in human]
        add("chair", "Кресло целиком", chair_bodies, metal + ", алюминий, резина, обивка и электроника")
        # Keep specific selection mappings instead of mapping every surface to chair.
        for key, part in self.groups.items():
            if key != "chair":
                for g in part["geoms"]:
                    self.geom_part[g] = key

    def list(self):
        return [dict(id=k, name=p["name"]) for k, p in self.groups.items()]

    def _bounds(self, d, g, origin, axes):
        m = self.m
        centre = axes.T @ (d.geom_xpos[g] - origin)
        R = axes.T @ d.geom_xmat[g].reshape(3, 3)
        size, kind = m.geom_size[g], m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]; adr = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
            pts = m.mesh_vert[adr:adr+count] @ R.T + centre
            return pts.min(axis=0), pts.max(axis=0)
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            return None
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = np.full(3, size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = np.abs(R[:, 2]) * size[1] + size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ext = np.abs(R[:, 2]) * size[1] + size[0] * np.sqrt(R[:, 0]**2 + R[:, 1]**2)
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            ext = np.sqrt((R**2) @ (size**2))
        else:
            ext = np.abs(R) @ size
        return centre - ext, centre + ext

    def inspect(self, d, key):
        part = self.groups[key]
        axes = d.xmat[self.frame].reshape(3, 3) if part["bodies"] else np.eye(3)
        origin = d.xpos[self.frame] if part["bodies"] else np.zeros(3)
        bounds = [v for g in part["geoms"] if (v := self._bounds(d, g, origin, axes)) is not None]
        dimensions = (np.max([b[1] for b in bounds], axis=0) - np.min([b[0] for b in bounds], axis=0)) * 1000 if bounds else None
        result = {k: part[k] for k in ("id", "name", "material", "note", "mass_kg")}
        result["dimensions_mm"] = dimensions.tolist() if dimensions is not None else None
        result["dimensions_note"] = "X × Y × Z в текущем положении, по видимой геометрии модели; без человека." if key == "chair" else "X × Y × Z в текущем положении, по геометрии модели."
        if key == "track_threshold_30mm":
            result["dimensions_note"] = "Длина по ходу × ширина × высота порога."
        return result
