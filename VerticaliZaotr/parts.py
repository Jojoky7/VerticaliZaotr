"""Inspection of physical assemblies, with geometry and mass from the loaded model.

Materials are design assumptions. Body inertials describe assemblies, so we do
not invent a separate mass for every decorative surface or infer material from colour.
"""
import math

import mujoco
import numpy as np


class Parts:
    def __init__(self, m, frame_mat, actuator, motor):
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
        self._engineering_specs(frame_mat, actuator, motor)

    def _engineering_specs(self, frame_mat, actuator, motor):
        """Keep catalogue values, calculations and unfinished design choices explicit."""
        self.motor, self.actuator = motor, actuator
        for part in self.groups.values():
            part.update(specs=[], sources=[], build_note="Габариты относятся к геометрии симуляции. Крепёж, допуски и рабочие чертежи этого узла не заданы.")

        def rows(key, *values, build_note=None):
            if key not in self.groups:
                return
            self.groups[key]["specs"] = [dict(label=label, value=str(value), basis=basis) for label, value, basis in values]
            if build_note:
                self.groups[key]["build_note"] = build_note

        for key in ("Motor150_Left", "Motor150_Right"):
            side = "левого" if key.endswith("Left") else "правого"
            custom = motor["key"] == "custom"
            rows(key,
                 ("Модель", motor["name"], "профиль"),
                 ("Тип", "Бесщёточный мотор постоянного тока с редуктором", "проект"),
                 ("Питание", f"{motor['voltage']:g} В", "профиль"),
                 ("Мощность на валу" if custom else "Номинальная мощность", f"{motor['power_w']:.1f} Вт", "расчёт" if custom else "каталог"),
                 ("Момент мотора", f"{motor['T']:g} Н·м", "профиль"),
                 ("Обороты мотора", f"{motor['n_rated']:g} об/мин под нагрузкой; {motor['n0']:g} без нагрузки", "профиль"),
                 ("Ток мотора", f"{motor['I']:g} А номинальный; {motor['I0']:g} А без нагрузки", "профиль"),
                 ("Редуктор в модели", f"{motor['ratio']:g}:1", "проект"),
                 ("КПД редуктора", f"{motor['eta']*100:g}%", "допущение"),
                 ("Выход редуктора", f"{motor['n_rated']/motor['ratio']:.1f} об/мин; {motor['wheel_torque']:.2f} Н·м на весь борт", "расчёт"),
                 ("Связь с колёсами", f"Один мотор для заднего и переднего колеса {side} борта", "схема"),
                 ("Вал и крепёж", "Диаметр вала, шпонка, отверстия и фланец не заданы", "нужно подобрать"),
                 build_note="Корпус в симуляции имеет условные размеры. Для изготовления нужны чертёж производителя, выходной вал и посадки, крепление редуктора, контроллер BLDC и тормоз. Передача на переднюю ось пока не выбрана; отдельного переднего мотора в проекте нет.")
            if not custom:
                self.groups[key]["sources"] = [dict(label="Характеристики производителя", url=motor["source"])]
                if not math.isclose(motor["ratio"], motor["catalog_ratio"]):
                    self.groups[key]["specs"].append(dict(label="Редуктор в каталоге", value=f"{motor['catalog_ratio']:g}:1; исполнение {motor['ratio']:g}:1 требует подтверждения", basis="каталог"))

        for key in ("drive_left", "drive_right", "front_left", "front_right"):
            part = self.groups[key]
            tyres = [g for g in part["geoms"] if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
            tyre = max(tyres, key=lambda g: self.m.geom_size[g, 0])
            radius, half_width = self.m.geom_size[tyre, :2]
            side = "левого" if key.endswith("left") else "правого"
            front = key.startswith("front_")
            rows(key,
                 ("Диаметр шины", f"{radius*2000:g} мм", "геометрия"),
                 ("Ширина шины", f"{half_width*2000:g} мм", "геометрия"),
                 ("Тип колеса", "Неповоротное ведущее; ось вращения поперёк кресла", "схема"),
                 ("Покрытие", "Резиновая шина; литое или пневматическое исполнение не выбрано", "проект"),
                 ("Обод и ступица", "Алюминиевый обод, стальная ступица; марки не заданы", "допущение"),
                 ("Привод", f"Общий мотор {side} борта через редуктор", "схема"),
                 ("Передняя ось" if front else "Передача вперёд", "Ремень / цепь / вал — вариант ещё не выбран", "нужно подобрать"),
                 ("Номинальный момент", f"{motor['wheel_torque']/2:.2f} Н·м на это колесо при равном делении", "расчёт"),
                 ("Обороты под нагрузкой", f"{motor['n_rated']/motor['ratio']:.1f} об/мин", "расчёт"),
                 ("Скорость по окружности", f"{motor['n_rated']/motor['ratio']*2*math.pi*radius*3.6/60:.2f} км/ч без проскальзывания", "расчёт"),
                 ("Протектор и метки", "12 поперечных рёбер и 3 метки на ободе, одна синяя", "визуализация"),
                 ("Посадка на ось", "Диаметр отверстия, подшипники, шпонка и крепёж не заданы", "нужно подобрать"),
                 build_note="Для изготовления нужно выбрать конкретную шину, её допустимую нагрузку и давление (если она пневматическая), обод, ступицу, подшипники и посадки. Видимые рёбра служат метками вращения; контакт с полом использует прежнюю гладкую модель шины.")
        for suffix in ("", "_right"):
            key = "actuator" + suffix
            rows(key,
                 ("Профиль", actuator["name"], "проект"),
                 ("Усилие одного", f"{actuator['unit_force']:g} Н наружу / {actuator['unit_retract_force']:g} Н внутрь", "профиль"),
                 ("Ход штока", "300 мм", "проект"),
                 ("Скорость без нагрузки", f"{actuator['speed']*1000:g} мм/с", "профиль"),
                 ("Длина между шарнирами", f"{actuator['retracted']*1000:g}–{(actuator['retracted']+.3)*1000:g} мм", "проект"),
                 ("Удержание", "Винтовой привод удерживает без питания", "допущение"),
                 ("Синхронизация", "Общая команда положения, отдельные пределы усилия; нужны датчики" if actuator["count"] == 2 else "Один привод", "схема"),
                 build_note="Сверить ход, сложенную длину, монтажные шарниры, допустимую боковую нагрузку, цикл работы, скорость под нагрузкой и датчики выбранного изделия. Размер корпуса здесь условный; длина между шарнирами задана отдельно.")
            if key in self.groups and actuator.get("source"):
                self.groups[key]["sources"] = [dict(label="Семейство привода; исполнение требует сверки", url=actuator["source"])]
        rows("frame", ("Материал рамы", "Сталь" if frame_mat == "steel" else "Алюминий 6061-T6", "проект"),
             ("Конструкция", "Трубчатая сварная рама; точные швы и крепёж не заданы", "проект"),
             build_note="Нужны чертежи труб и кронштейнов, марки металла, сварные соединения, крепёж и проверка прочности. Масса рамы включает закреплённые узлы по данным симуляции.")
        rows("Battery_24V_20Ah", ("Химия", "LiFePO₄", "проект"), ("Напряжение", "25,6 В", "проект"),
             ("Ёмкость", "20 А·ч / 512 Вт·ч", "расчёт"),
             ("Защита", "BMS, предохранитель, разъёмы и допустимый ток не выбраны", "нужно подобрать"))
        rows("seat", ("Конструкция", "Металлическая основа, фанера, пеноматериал и обивка", "проект"),
             ("Диапазон подъёма", "0–75°", "схема"))
        rows("back", ("Конструкция", "Металлическая основа, мягкая обивка, подлокотники", "проект"))
        rows("lower_link", ("Назначение", "Замыкание четырёхзвенного механизма сиденья и спинки", "схема"),
             ("Шарниры", "Оси, втулки и допуски не заданы", "нужно подобрать"))
        for side in ("left", "right"):
            rows("front_fork_" + side, ("Тип", "Неповоротная опора ведущего переднего колеса", "схема"),
                 ("Крепление", "Ось, подшипники и крепёж требуют подбора", "нужно подобрать"))
        rows("chair", ("Схема движения", "Два мотора; каждый приводит два колеса своего борта", "схема"),
             ("Поворот", "Разная скорость левого и правого борта", "схема"),
             ("Подъём", f"{actuator['count']} актуатор(а)", "схема"),
             build_note="Это компоновочная модель. Карточки показывают размеры модели, выбранные параметры и незавершённые пункты для изготовления. Полного комплекта производственных чертежей пока нет.")

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
        result = {k: part[k] for k in ("id", "name", "material", "note", "mass_kg", "sources", "build_note")}
        result["specs"] = list(part["specs"])
        if key in ("drive_left", "drive_right", "front_left", "front_right"):
            j = self.m.joint(key + "_spin").dofadr[0]
            result["specs"].append(dict(label="Сейчас", value=f"{d.qvel[j]*60/(2*math.pi):.1f} об/мин; {d.qfrc_actuator[j]*d.qvel[j]:.1f} Вт на оси", basis="телеметрия"))
        if key.startswith("Motor150_"):
            side = "left" if key.endswith("Left") else "right"
            dofs = [self.m.joint(prefix + side + "_spin").dofadr[0] for prefix in ("drive_", "front_")]
            power = sum(float(d.qfrc_actuator[j]*d.qvel[j]) for j in dofs)
            result["specs"].append(dict(label="Сейчас на борту", value=f"{power:.1f} Вт на двух осях", basis="телеметрия"))
        result["dimensions_mm"] = dimensions.tolist() if dimensions is not None else None
        result["dimensions_note"] = "X × Y × Z в текущем положении, по видимой геометрии модели; без человека." if key == "chair" else "X × Y × Z в текущем положении, по геометрии модели."
        if key == "track_threshold_30mm":
            result["dimensions_note"] = "Длина по ходу × ширина × высота порога."
        return result
