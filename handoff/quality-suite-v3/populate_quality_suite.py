#!/usr/bin/env python3
"""Populate an existing quality-dimensions-llm project with reproducible examples.

Python 3.10+; standard library only; no model downloads or inference.
Usage: python3 populate_quality_suite.py --project-root /path/to/project
       python3 populate_quality_suite.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

VERSION = "3.0.0"
SUITE = "quality_suite_v3"
BASE = f"data/{SUITE}"
AUTHOR = "Original assistant-authored experimental fixtures; not collected observations."
FAMILIES = {
    "plain": ("train", "words"),
    "numeric": ("train", "numeric"),
    "paraphrase": ("heldout", "alternate"),
    "reordered": ("heldout", "alternate"),
}


@dataclass(frozen=True)
class Quality:
    key: str
    domain: str
    quantity: str
    unit: str
    values: tuple
    words: tuple
    alternate: tuple
    anchors: tuple
    higher: str
    lower: str
    comparison_subject: str
    note: str
    numeric_sweep: bool = False
    log_sweep: bool = False
    extrapolate: bool = False


def Q(key, domain, quantity, unit, values, words, alternate, anchors,
      higher, lower, comparison_subject, note, sweep=False, log=False, extrap=False):
    return Quality(key, domain, quantity, unit, tuple(values), tuple(words.split("|")),
                   tuple(alternate.split("|")), tuple(tuple(g.split("|")) for g in anchors),
                   higher, lower, comparison_subject, note, sweep, log, extrap)


QUALITIES = [
    Q("size", "spatial", "longest linear extent", "cm", [1, 5, 15, 50, 150],
      "is tiny|is small|is medium-sized|is large|is enormous",
      "has a minute overall extent|has a modest overall extent|has an intermediate overall extent|has a substantial overall extent|has a vast overall extent",
      ["salt grain|ant|pea|blueberry|button", "strawberry|walnut|apple|teacup|tennis ball",
       "loaf of bread|book|rabbit|basketball|brick", "chair|bicycle|dog|armchair|adult person",
       "elephant|shipping container|bus|airplane|building"],
      "longer", "shorter", "the longest extent of {name}",
      "Controlled numeric size means longest linear extent, not mass, volume, visual angle or human typicality. Lexical size ranks are provisional and may overlap.", True, True, True),
    Q("mass", "mechanical", "mass", "g", [1, 10, 100, 1000, 10000],
      "is extremely light|is lightweight|has an intermediate mass|is heavy|is extremely heavy",
      "has very little mass|has a low mass|has a moderate mass|has a high mass|has a very high mass",
      ["feather|paperclip|postage stamp|rice grain|dry leaf", "pencil|coin|key|eraser|teabag",
       "apple|smartphone|mug|orange|bar of soap", "laptop|brick|dictionary|bowling ball|kettlebell",
       "refrigerator|motorcycle|car|elephant|locomotive"],
      "greater", "smaller", "the mass of {name}",
      "Mass is distinct from size; ordinary noun masses vary greatly. No real measurements are supplied.", True, True, True),
    Q("temperature", "thermal", "temperature", "degrees Celsius", [0, 10, 20, 40, 80],
      "is very cold|is cool|is at an intermediate temperature|is warm|is very hot",
      "has a very low temperature|has a low temperature|has a moderate temperature|has an elevated temperature|has a very high temperature",
      ["dry ice|ice cube|snow|frozen lake|icicle", "chilled water|refrigerated milk|cold cellar|cool stream|winter air",
       "room-temperature water|indoor air|wooden desk|room-temperature apple|unheated ceramic tile",
       "warm bath|sunlit pavement|fresh tea|heating pad|warm radiator",
       "boiling water|oven interior|campfire|molten lava|blast furnace"],
      "higher", "lower", "the temperature of {name}",
      "Celsius values are assigned fictional specifications. Qualitative words provide only designed ordinal categories, not exact temperatures.", True, False, True),
    Q("speed", "motion", "speed", "metres per second", [0, 1, 5, 20, 100],
      "is motionless|moves slowly|moves at an intermediate speed|moves quickly|moves extremely quickly",
      "does not move|travels at a low speed|travels at a moderate speed|travels at a high speed|travels at a very high speed",
      ["parked car|stationary statue|resting stone|stopped clock hand|anchored boat",
       "snail|tortoise|crawling baby|drifting leaf|walking pedestrian",
       "jogger|cyclist|swimmer|running dog|rolling skateboard",
       "galloping horse|moving train|motorcycle on a highway|cheetah|speedboat",
       "jet aircraft|racing car|rocket|meteor|spacecraft"],
      "higher", "lower", "the speed of {name}",
      "Lexical examples include state descriptions and therefore possible lexical shortcuts. Numeric scenes specify a common reference frame.", True, False, False),
    Q("brightness", "visual", "emitted luminance", "candela per square metre", [1, 10, 100, 1000, 10000],
      "looks extremely dim|looks dim|has an intermediate brightness|looks bright|looks extremely bright",
      "has very low apparent brightness|has low apparent brightness|has moderate apparent brightness|has high apparent brightness|has very high apparent brightness",
      ["unlit cave|black cloth in shade|moonless sky|unlit screen|dark cupboard",
       "candlelit room|night-light|twilight sky|dim hallway|glowing ember",
       "cloudy sky|desk lamp|computer screen|lit kitchen|overcast courtyard",
       "sunlit white wall|stage spotlight|car headlight|bright shop window|sunlit snow",
       "sun's disc|camera flash|welding arc|lightning flash|floodlight"],
      "higher", "lower", "the luminance of {name}",
      "Luminance is a physical proxy; brightness depends on adaptation and context. Lexical examples are NOT a calibrated common photometric scale.", True, True, True),
    Q("loudness", "auditory", "sound pressure level at the same listener position", "dB SPL", [20, 40, 60, 80, 100],
      "produces a barely audible sound|produces a quiet sound|produces a moderately loud sound|produces a loud sound|produces an extremely loud sound",
      "sounds very faint|sounds soft|sounds intermediate in loudness|sounds powerful|sounds deafening",
      ["falling snow|distant rustle|soft breathing|ticking wristwatch|faint hum",
       "whisper|turning book page|gentle rain|quiet fan|soft footsteps",
       "conversation|running tap|piano practice|busy cafe|television",
       "vacuum cleaner|shouting crowd|passing truck|power drill|barking dog",
       "thunderclap|jet takeoff|stadium roar|fire alarm|rock concert"],
      "higher", "lower", "the sound pressure level produced by {name}",
      "SPL is not perceived loudness. Controlled single-property scenes specify a common listener position and tone frequency; lexical sound examples do not.", True, False, False),
    Q("pitch", "auditory", "frequency of its pure tone", "Hz", [110, 220, 440, 880, 1760],
      "produces a very low-pitched tone|produces a low-pitched tone|produces a mid-pitched tone|produces a high-pitched tone|produces a very high-pitched tone",
      "emits a very deep tone|emits a deep tone|emits a middle-register tone|emits a treble tone|emits an upper-register tone",
      ["bass drum|low organ pedal|tuba|bass guitar|distant rumble",
       "cello|baritone voice|bassoon|low piano note|bull's bellow",
       "conversational voice|viola|guitar middle string|clarinet middle register|middle piano note",
       "flute|soprano voice|violin upper register|birdsong|whistling kettle",
       "piccolo|mosquito whine|high electronic beep|squeaky hinge|bat chirp"],
      "higher", "lower", "the pure-tone frequency of {name}",
      "Frequency is stated for controlled pure tones; complex lexical sounds have broad, variable spectra. Log-frequency analysis is a separately declared option.", True, True, True),
    Q("roughness", "surface", "roughness on the fictional five-level scale", "scale points", [1, 2, 3, 4, 5],
      "has a perfectly smooth surface|has a slightly textured surface|has a moderately textured surface|has a rough surface|has an extremely rough surface",
      "feels silky smooth|feels almost smooth|feels somewhat uneven|feels coarse|feels very coarse and irregular",
      ["polished glass|silk|mirror|glazed porcelain|smooth ice",
       "satin|plastic ruler|polished wood|fine paper|smooth leather",
       "cotton cloth|cardboard|unpolished wood|woven linen|orange peel",
       "tree bark|sandpaper|gravel|coarse fabric|rough stone",
       "jagged rock|coarse rasp|broken concrete|sharp lava rock|very coarse sandpaper"],
      "rougher", "smoother", "{name}",
      "Ordinal descriptions only; roughness is not identical to friction, hardness or bump wavelength."),
    Q("hardness", "mechanical", "indentation resistance on the fictional five-level scale", "scale points", [1, 2, 3, 4, 5],
      "is extremely easy to indent|is easy to indent|offers moderate resistance to indentation|is difficult to indent|is extremely difficult to indent",
      "yields under very light pressure|yields under light pressure|yields under moderate pressure|requires strong pressure to dent|strongly resists being dented",
      ["whipped cream|soft foam|fresh dough|cotton wool|gelatin dessert",
       "rubber eraser|soft clay|candle wax|ripe banana|cork",
       "leather|cardboard|pine wood|firm rubber|packed soil",
       "hardwood|brick|aluminium plate|ceramic tile|bone",
       "steel block|granite|quartz|hardened tool steel|diamond"],
      "harder", "softer", "{name}",
      "Fictional ordinal indentation resistance, not Mohs hardness, stiffness, strength or brittleness. Lexical ranks are provisional."),
    Q("elasticity", "mechanical", "recovery after the same small deformation", "percent", [0, 25, 50, 75, 100],
      "retains all of the imposed deformation|recovers little of its original shape|recovers half of its original shape|recovers most of its original shape|fully recovers its original shape",
      "does not spring back|barely springs back|partly springs back|mostly springs back|completely springs back",
      ["molded clay|kneaded dough|crumpled foil|bent paperclip|putty",
       "damp cardboard|creased paper|soft wax|compressed sand|flattened dough",
       "worn sponge|memory foam|soft leather|felt pad|flexible packing foam",
       "cork stopper|silicone sheet|springy foam|flexible plastic strip|wool cushion",
       "rubber band|steel spring|bungee cord|elastic fabric|rubber ball"],
      "greater", "smaller", "the elastic recovery of {name}",
      "Recovery under a specified deformation is distinct from elastic modulus. Ordinary behavior depends on loading history; lexical ranks are examples only.", True, False, False),
    Q("sweetness", "gustatory", "Mira's stated sweetness rating on the fictional five-level scale", "scale points", [1, 2, 3, 4, 5],
      "tastes unsweet to Mira|tastes faintly sweet to Mira|tastes moderately sweet to Mira|tastes very sweet to Mira|tastes intensely sweet to Mira",
      "has no sweetness for Mira|has a hint of sweetness for Mira|has a noticeable sweetness for Mira|has a strong sweetness for Mira|has an overwhelming sweetness for Mira",
      ["plain water|unsweetened tea|black coffee|celery|cucumber",
       "plain milk|carrot|oatmeal|plain bread|unsweetened yogurt",
       "apple|banana|sweetcorn|strawberry|pear",
       "ripe mango|cake|sweetened yogurt|milk chocolate|fruit jam",
       "honey|sugar syrup|hard candy|marshmallow|icing sugar"],
      "sweeter", "less sweet", "the taste of {name} for Mira",
      "Taste and recipes vary. The named observer and scale are fictional; no participant ratings exist."),
    Q("pleasantness", "affective", "Mira's stated pleasantness rating", "scale points", [-2, -1, 0, 1, 2],
      "is very unpleasant for Mira|is unpleasant for Mira|is neither pleasant nor unpleasant for Mira|is pleasant for Mira|is very pleasant for Mira",
      "feels strongly aversive to Mira|feels somewhat aversive to Mira|feels affectively neutral to Mira|feels enjoyable to Mira|feels deeply enjoyable to Mira",
      ["toothache|bereavement|humiliation|severe loneliness|nausea",
       "traffic delay|minor argument|paper cut|boredom|unwanted noise",
       "blank form|plain wall|neutral announcement|empty cardboard box|ordinary timetable",
       "friendly conversation|warm bath|favorite meal|sunny walk|pleasant music",
       "joyful reunion|deep relief|loving embrace|major achievement|wonderful surprise"],
      "more pleasant", "less pleasant", "Mira's experience of {name}",
      "Observer-, culture- and situation-dependent candidate; not an objective property assigned to every noun."),
    Q("arousal", "affective", "Mira's stated arousal rating on the fictional five-level scale", "scale points", [1, 2, 3, 4, 5],
      "leaves Mira very drowsy|leaves Mira calm|leaves Mira moderately alert|leaves Mira highly alert|leaves Mira extremely activated",
      "barely activates Mira|slightly activates Mira|moderately activates Mira|strongly activates Mira|intensely activates Mira",
      ["sleepy afternoon|quiet rest|slow breathing|drowsiness|gentle lullaby",
       "calm reading|peaceful garden|soft music|relaxed conversation|slow walk",
       "ordinary meeting|light exercise|interesting lecture|friendly game|routine shopping",
       "lively celebration|competitive match|urgent deadline|exciting news|tense argument",
       "panic|intense excitement|sudden alarm|thrilling climax|extreme surprise"],
      "higher", "lower", "Mira's arousal while experiencing {name}",
      "Arousal is not pleasantness. Synthetic scene ratings are not clinical observations."),
    Q("danger", "evaluative", "stated interaction danger on the fictional five-level scale", "scale points", [1, 2, 3, 4, 5],
      "is harmless to interact with in this scene|poses a minor danger in this scene|poses a moderate danger in this scene|poses a serious danger in this scene|poses an extreme danger in this scene",
      "presents no stated threat|presents a small stated threat|presents a medium stated threat|presents a major stated threat|presents a very severe stated threat",
      ["soft cushion|folded towel|paper bookmark|plush toy|empty notebook",
       "paper edge|uneven paving stone|warm cup|small splinter|mildly slippery floor",
       "steep staircase|sharp kitchen knife|busy road|deep pond|hot pan",
       "cliff edge|unstable ladder|large aggressive animal|fast-moving traffic|exposed electrical wire",
       "wildfire|avalanche|collapsing building|major flood|violent tornado"],
      "more dangerous", "less dangerous", "interacting with {name}",
      "An evaluative candidate depending on agent, action and circumstances; fictional ratings are not real risk estimates or safety advice."),
]
BY_KEY = {q.key: q for q in QUALITIES}
HUES = [(0, "red"), (30, "orange"), (60, "yellow"), (90, "yellow-green"),
        (120, "green"), (150, "green-cyan"), (180, "cyan"), (210, "azure"),
        (240, "blue"), (270, "violet"), (300, "magenta"), (330, "rose")]
NAMES = "Kalo Neri Pavo Sumi Leto Vanu Riko Tavi Melo Zuri Dema Feno Gavi Huno Jari Lumi Navo Peli Rumi Savo".split()


def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)


def pretty(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest(x):
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def number(x):
    return f"{x:.8g}"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def csv_text(rows, fields):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: dumps(row[k]) if isinstance(row.get(k), (dict, list, tuple)) else row.get(k, "") for k in fields})
    return f.getvalue()


def legacy_info(root):
    p = root / "data/concepts.csv"
    if not p.is_file():
        return [], set()
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        names = {row[k].strip().casefold() for row in reader for k in ("lemma", "noun", "name", "word") if row.get(k)}
    return fields, names


def lexical_split(lemma, seed, previous):
    if lemma.casefold() in previous:
        return "train"
    n = int(digest(f"{seed}|lexical|{lemma.casefold()}")[:8], 16) % 10
    return "train" if n < 6 else "validation" if n < 8 else "test"


def hue_targets(degrees):
    angle = math.radians(degrees % 360)
    return {"hue_sin": round(math.sin(angle), 12), "hue_cos": round(math.cos(angle), 12)}


def frame(facts, focus, family):
    lead = {"plain": "In this fictional scene", "numeric": "Fictional laboratory specification",
            "paraphrase": "Imagine the following invented situation", "reordered": "Here is an invented observation record"}[family]
    return f"{lead}: {facts}\nFocus on {focus}."


def clause(q, name, index, family):
    mode = FAMILIES[family][1]
    if mode == "numeric":
        return f"The {q.quantity} for {name} is {number(q.values[index])} {q.unit}."
    words = q.words if mode == "words" else q.alternate
    return f"{name} {words[index]}."


def domain_setup(q):
    if q.key == "loudness":
        return "All sounds here are 440 Hz pure tones of equal duration at the same listener position. "
    if q.key == "pitch":
        return "All sounds here are pure tones of equal duration, at 60 dB SPL at the same listener position. "
    if q.key == "speed":
        return "All speeds here use the same reference frame. "
    return ""


class Builder:
    def __init__(self, seed=20260928, entity_count=20, previous=None):
        self.seed, self.previous = seed, previous or set()
        order = list(range(entity_count))
        random.Random(seed).shuffle(order)
        nt = int(0.6 * entity_count)
        nv = int(0.2 * entity_count)
        split = {n: "train" if i < nt else "validation" if i < nt + nv else "test" for i, n in enumerate(order)}
        self.entities = [{"entity_id": f"entity_{i:02d}", "name": NAMES[i], "foil": NAMES[i] + "x",
                          "entity_split": split[i], "tag": "blue" if i % 2 else "red"} for i in range(entity_count)]
        self.cases, self.by_id, self.contrasts, self.comparisons, self.triplets = [], {}, [], [], []
        self.concepts, self.labels, self.factorial_domains = {}, [], {}

    def add(self, kind, key, group, split, family, focus, facts, targets, **extra):
        text = frame(facts, focus, family)
        case_id = kind + "_" + digest(key)[:20]
        if case_id in self.by_id:
            raise ValueError(f"Duplicate case identity: {key}")
        start = text.rfind(focus)
        row = dict(case_id=case_id, case_type=kind, entity_group_id=group, entity_split=split,
                   template_id=family, template_split=FAMILIES[family][0], text=text,
                   readout_entity=focus, readout_char_span=[start, start + len(focus)],
                   targets=targets, quality_ids=list(targets), label_kind="synthetic_ordinal",
                   specified_values={}, combination_split="train", provenance=AUTHOR,
                   evaluation_status="exploratory_constructed_fixture")
        row.update(extra)
        self.cases.append(row)
        self.by_id[case_id] = row
        return case_id

    def contrast(self, low, high, quality, kind="increase", changed_axis=None):
        a, b = self.by_id[low], self.by_id[high]
        self.contrasts.append(dict(contrast_id="contrast_" + digest(low + high + quality + kind)[:20],
                                  low_case_id=low, high_case_id=high, quality_id=quality, kind=kind,
                                  changed_axis=changed_axis or quality, entity_group_id=a["entity_group_id"],
                                  entity_split=a["entity_split"], template_id=a["template_id"],
                                  template_split=a["template_split"], combination_split=("test" if "test" in [a["combination_split"], b["combination_split"]] else "train"),
                                  expected_relation="invariant" if "invariant" in kind else "higher"))

    def lexical(self):
        for q in QUALITIES:
            for level, words in enumerate(q.anchors):
                for lemma in words:
                    group = "lex_" + slug(lemma)
                    split = lexical_split(lemma, self.seed, self.previous)
                    self.concepts.setdefault(group, dict(concept_id=group, lemma=lemma, synonym_group=group,
                        category="lexical_example", source_kind="authored_demo", entity_split=split,
                        previous_pilot_overlap=lemma.casefold() in self.previous))
                    self.labels.append(dict(concept_id=group, quality_id=q.key, ordinal_score=level + 1,
                        label_kind="demo_ordinal", label_source=AUTHOR,
                        notes="Provisional illustrative ranking, not a human norm or measurement."))
                    for family in ("plain", "paraphrase"):
                        facts = f"The item under discussion is {lemma}." if family == "plain" else f"The discussion concerns {lemma}."
                        self.add("lexical", f"{q.key}|{group}|{family}", group, split, family, lemma,
                                 facts, {q.key: level / 4}, quality_ids=[q.key], label_kind="demo_ordinal",
                                 level_index=level, prior_pilot_overlap=lemma.casefold() in self.previous)

    def controlled(self):
        for q, e, family in itertools.product(QUALITIES, self.entities, FAMILIES):
            name, foil = e["name"], e["foil"]
            common = domain_setup(q) + f"{name} and {foil} are invented test items. {name} carries a {e['tag']} tag. "
            ids = []
            for level in range(5):
                facts = common + clause(q, name, level, family)
                cid = self.add("graded", f"{q.key}|{name}|{family}|{level}", e["entity_id"], e["entity_split"], family, name,
                               facts, {q.key: level / 4}, quality_ids=[q.key], level_index=level, condition="target_level",
                               specified_values={q.key: q.values[level]} if family == "numeric" else {},
                               label_kind="synthetic_stated_numeric_order" if family == "numeric" else "synthetic_ordinal")
                ids.append(cid)
            for i in range(4):
                self.contrast(ids[i], ids[i + 1], q.key, "adjacent_increase")
            self.contrast(ids[0], ids[4], q.key, "full_range_increase")
            # Controls retain the subject's middle level. No imaginary human labels.
            controls = {
                "irrelevant": clause(q, name, 2, family) + " The record was filed on a Tuesday.",
                "distractor_low": clause(q, name, 2, family) + " " + clause(q, foil, 0, family),
                "distractor_high": clause(q, name, 2, family) + " " + clause(q, foil, 4, family),
            }
            control_ids = {}
            for condition, body in controls.items():
                cid = self.add("controls", f"{q.key}|{name}|{family}|{condition}", e["entity_id"], e["entity_split"], family,
                               name, common + body, {q.key: .5}, quality_ids=[q.key], level_index=2, condition=condition,
                               specified_values={q.key: q.values[2]} if family == "numeric" else {},
                               label_kind="synthetic_stated_numeric_order" if family == "numeric" else "synthetic_ordinal")
                control_ids[condition] = cid
                self.contrast(ids[2], cid, q.key, condition + "_invariant")
            self.contrast(control_ids["distractor_low"], control_ids["distractor_high"], q.key, "distractor_direction_invariant")
            for level, denied in [(0, 4), (4, 0)]:
                denial = clause(q, name, denied, family).rstrip(".")
                body = f'The claim "{denial}" is false. In fact, {clause(q, name, level, family)}'
                cid = self.add("controls", f"{q.key}|{name}|{family}|negated_{level}", e["entity_id"], e["entity_split"], family,
                               name, common + body, {q.key: level / 4}, quality_ids=[q.key], level_index=level,
                               condition="negated_foil", specified_values={q.key: q.values[level]} if family == "numeric" else {},
                               label_kind="synthetic_stated_numeric_order" if family == "numeric" else "synthetic_ordinal")
                self.contrast(ids[level], cid, q.key, "negation_invariant")

    def binding(self):
        for q, e, family in itertools.product(QUALITIES, self.entities, ("plain", "paraphrase")):
            lookup = {}
            for assignment, reverse, query_foil in itertools.product((0, 1), (False, True), (False, True)):
                levels = [0, 4] if assignment == 0 else [4, 0]
                actors = [e["name"], e["foil"]]
                facts = [clause(q, actor, level, family) for actor, level in zip(actors, levels)]
                if reverse:
                    facts.reverse()
                which = int(query_foil)
                cid = self.add("binding", f"{q.key}|{e['name']}|{family}|{assignment}|{reverse}|{query_foil}",
                    e["entity_id"], e["entity_split"], family, actors[which], domain_setup(q) + " ".join(facts),
                    {q.key: levels[which] / 4}, quality_ids=[q.key], level_index=levels[which],
                    assignment=assignment, reversed_clause_order=reverse, query_foil=query_foil,
                    actor_levels=dict(zip(actors, levels)), condition="same_words_different_binding")
                lookup[assignment, reverse, query_foil] = cid
            for reverse, query in itertools.product((False, True), repeat=2):
                a, b = lookup[0, reverse, query], lookup[1, reverse, query]
                low, high = sorted((a, b), key=lambda x: self.by_id[x]["targets"][q.key])
                self.contrast(low, high, q.key, "binding_increase")
            for assignment, query in itertools.product((0, 1), (False, True)):
                self.contrast(lookup[assignment, False, query], lookup[assignment, True, query], q.key, "clause_order_invariant")

    def numeric(self):
        conversions = {"size": (lambda v: v / 100, "metres"), "mass": (lambda v: v / 1000, "kg"),
            "temperature": (lambda v: v * 9 / 5 + 32, "degrees Fahrenheit"),
            "speed": (lambda v: v * 3.6, "km per hour"), "pitch": (lambda v: v / 1000, "kHz")}
        for q, e in itertools.product([q for q in QUALITIES if q.numeric_sweep], self.entities):
            lo, hi = q.values[0], q.values[-1]
            positions = [i / 8 for i in range(9)] + ([-.125, 1.125] if q.extrapolate else [])
            ids = []
            for position in sorted(positions):
                value = math.exp(math.log(lo) + position * (math.log(hi) - math.log(lo))) if q.log_sweep else lo + position * (hi - lo)
                value = float(number(value))  # label matches the rounded value actually displayed
                given = domain_setup(q) + f"The {q.quantity} for {e['name']} is {number(value)} {q.unit}."
                role = "extrapolation" if not 0 <= position <= 1 else "seen_value" if any(math.isclose(value, v, rel_tol=1e-7, abs_tol=1e-7) for v in q.values) else "interpolation"
                cid = self.add("numeric", f"{q.key}|{e['name']}|{position}", e["entity_id"], e["entity_split"], "numeric", e["name"],
                               given, {q.key + "__value": value}, quality_ids=[q.key], specified_values={q.key: value},
                               label_kind="synthetic_stated_numeric", value_role=role, condition="numeric_sweep",
                               analysis_partition="challenge_only_not_for_fitting")
                ids.append(cid)
                if q.key in conversions:
                    fn, unit = conversions[q.key]
                    displayed = float(number(fn(value)))
                    alt = self.add("numeric", f"{q.key}|{e['name']}|{position}|converted", e["entity_id"], e["entity_split"], "numeric", e["name"],
                                   domain_setup(q) + f"The {q.quantity} for {e['name']} is {number(displayed)} {unit}.",
                                   {q.key + "__value": value}, quality_ids=[q.key], specified_values={q.key: value},
                                   label_kind="synthetic_stated_numeric", value_role=role, condition="unit_conversion",
                                   displayed_value=displayed, displayed_unit=unit,
                                   analysis_partition="challenge_only_not_for_fitting")
                    self.contrast(cid, alt, q.key + "__value", "unit_conversion_invariant")
            for a, b in zip(ids, ids[1:]):
                self.contrast(a, b, q.key + "__value", "numeric_adjacent_increase")

    def hue(self):
        for e, family in itertools.product(self.entities, ("plain", "numeric", "paraphrase", "reordered")):
            ids = {}
            for degrees, color in HUES + ([(360, "red")] if family == "numeric" else []):
                if family == "numeric":
                    facts = f"The display patch named {e['name']} has HSV hue {degrees} degrees, saturation 100 percent, and value 100 percent."
                else:
                    facts = f"The display patch named {e['name']} is assigned the standard HSV color label {color}, with full saturation and value."
                cid = self.add("hue", f"{e['name']}|{family}|{degrees}", e["entity_id"], e["entity_split"], family, e["name"],
                    facts, hue_targets(degrees), quality_ids=["hue"], specified_values={"hue_degrees": degrees},
                    raw_hue_degrees=degrees, canonical_hue_degrees=degrees % 360, label_kind="synthetic_color_convention",
                    condition="circular_hue")
                ids[degrees] = cid
            if 360 in ids:
                self.contrast(ids[0], ids[360], "hue", "circular_wrap_invariant")
            for degrees, _ in HUES:
                self.triplets.append(dict(triplet_id="hue_triplet_" + digest(ids[degrees])[:20],
                    anchor_case_id=ids[degrees], near_case_id=ids[(degrees + 30) % 360], far_case_id=ids[(degrees + 180) % 360],
                    geometry="circular_HSV_hue", near_distance_degrees=30, far_distance_degrees=180,
                    entity_group_id=e["entity_id"], entity_split=e["entity_split"], template_id=family,
                    template_split=FAMILIES[family][0]))

    def factorial(self):
        domains = {"extent_mass": ["size", "mass"], "shape": ["length", "width", "height"],
                   "surface": ["roughness", "hardness", "elasticity"], "sound": ["pitch", "loudness"],
                   "affect": ["pleasantness", "arousal", "danger"], "colour": ["hue", "saturation", "value"],
                   "motion_thermal": ["speed", "temperature"]}
        aux = {"length": [5, 15, 45], "width": [5, 15, 45], "height": [5, 15, 45],
               "saturation": [25, 60, 100], "value": [25, 60, 100]}
        for domain, axes in domains.items():
            sizes = [6 if axis == "hue" else 3 for axis in axes]
            combos = list(itertools.product(*(range(n) for n in sizes)))
            self.factorial_domains[domain] = dict(axes=axes, grid_shape=sizes, cells=len(combos),
                heldout_rule="weighted index sum modulo 5 equals 1", synthetic=True)
            for e, family in itertools.product(self.entities, ("numeric", "reordered")):
                ids = {}
                for indices in combos:
                    values, targets, statements = {}, {}, []
                    for axis, index in zip(axes, indices):
                        if axis in BY_KEY:
                            q = BY_KEY[axis]
                            val = q.values[index + 1]
                            values[axis] = val
                            targets[axis] = (index + 1) / 4
                            statements.append(f"Its {q.quantity} is {number(val)} {q.unit}.")
                        elif axis == "hue":
                            values[axis] = index * 60
                            targets.update(hue_targets(index * 60))
                            statements.append(f"Its HSV hue is {index * 60} degrees.")
                        else:
                            values[axis] = aux[axis][index]
                            targets[axis] = index / 2
                            unit = "cm" if axis in ("length", "width", "height") else "percent"
                            label = "HSV " + axis if axis in ("saturation", "value") else axis
                            statements.append(f"Its {label} is {number(values[axis])} {unit}.")
                    derived = {}
                    if domain == "shape":
                        derived["volume_cm3"] = math.prod(values[a] for a in axes)
                    if family == "reordered":
                        statements.reverse()
                    facts = f"{e['name']} is a fictional {'rectangular block' if domain == 'shape' else 'test item'}. " + " ".join(statements)
                    combination_split = "test" if sum((j + 1) * n for j, n in enumerate(indices)) % 5 == 1 else "train"
                    ids[indices] = self.add("factorial", f"{domain}|{e['name']}|{family}|{indices}", e["entity_id"], e["entity_split"], family,
                        e["name"], facts, targets, quality_ids=axes, specified_values=values, derived_values=derived,
                        primitive_axes=axes, factorial_domain=domain, factorial_indices=list(indices),
                        combination_split=combination_split, label_kind="synthetic_factorial_specification",
                        condition="independently_assigned_attributes")
                for indices in combos:
                    for j, axis in enumerate(axes):
                        if axis == "hue" or indices[j] == sizes[j] - 1:
                            continue
                        next_indices = list(indices)
                        next_indices[j] += 1
                        self.contrast(ids[indices], ids[tuple(next_indices)], axis, "factorial_one_axis_increase", axis)

    def behavior(self):
        for q, e, family, level in itertools.product(QUALITIES, self.entities, ("plain", "paraphrase", "numeric"), (0, 4)):
            facts = domain_setup(q) + clause(q, e["name"], level, family) + " " + clause(q, e["foil"], 2, family)
            for reverse in (False, True):
                subject, reference = (e["foil"], e["name"]) if reverse else (e["name"], e["foil"])
                higher = (level > 2) != reverse
                stem = q.comparison_subject.format(name=subject)
                base = dict(quality_id=q.key, entity_group_id=e["entity_id"], entity_split=e["entity_split"],
                            template_id=family, template_split=FAMILIES[family][0], level_index=level,
                            reversed_comparison=reverse, expected_relation="higher" if higher else "lower",
                            label_kind="synthetic_order_from_stated_context", provenance=AUTHOR)
                natural = f"In this fictional scene, {facts}\nCompared with {reference}, {stem} is"
                self.comparisons.append(dict(base, comparison_id="cmp_" + digest(natural)[:20],
                    format="natural_continuation", prefix=natural, candidates=[" " + q.higher, " " + q.lower],
                    candidate_meanings=["higher", "lower"], correct_index=0 if higher else 1,
                    answer_mapping="semantic_words", option_order="not_applicable"))
                for swap, print_reverse in itertools.product((False, True), repeat=2):
                    meanings = ["lower", "higher"] if swap else ["higher", "lower"]
                    description = {"higher": q.higher, "lower": q.lower}
                    options = [f"A = {description[meanings[0]]}", f"B = {description[meanings[1]]}"]
                    if print_reverse:
                        options.reverse()
                    prompt = f"In this fictional scene, {facts}\nCompared with {reference}, {stem} is which of the following?\nOptions: {'; '.join(options)}.\nAnswer:"
                    self.comparisons.append(dict(base, comparison_id="cmp_" + digest(prompt)[:20],
                        format="AB", prefix=prompt, candidates=[" A", " B"], candidate_meanings=meanings,
                        correct_index=meanings.index(base["expected_relation"]), answer_mapping="swapped" if swap else "original",
                        option_order="B_first" if print_reverse else "A_first"))

    def build(self):
        self.lexical()
        self.controlled()
        self.binding()
        self.numeric()
        self.hue()
        self.factorial()
        self.behavior()
        return self


def validate(builder):
    """Check referential integrity and the experimental changes, not model success."""
    b = builder
    if len(b.cases) != len(b.by_id):
        raise ValueError("Duplicate case IDs")
    groups = defaultdict(set)
    for row in b.cases:
        groups[row["entity_group_id"]].add(row["entity_split"])
        start, end = row["readout_char_span"]
        if row["text"][start:end] != row["readout_entity"] or not row["text"].endswith(row["readout_entity"] + "."):
            raise ValueError("Readout span does not identify the final contextualized entity")
        if row["template_split"] != FAMILIES[row["template_id"]][0]:
            raise ValueError("Template split changed")
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in row["targets"].values()):
            raise ValueError("Nonfinite or nonnumeric target")
        if row["label_kind"] == "demo_ordinal" and row["case_type"] != "lexical":
            raise ValueError("Label provenance mixed")
        if row.get("prior_pilot_overlap") and row["entity_split"] != "train":
            raise ValueError("Known original-pilot noun used as a new held-out noun")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("An entity group crosses splits")
    for row in b.contrasts:
        a, z = b.by_id[row["low_case_id"]], b.by_id[row["high_case_id"]]
        if (a["entity_group_id"], a["entity_split"], a["template_id"]) != (z["entity_group_id"], z["entity_split"], z["template_id"]):
            raise ValueError("Unpaired contrast")
        q = row["quality_id"]
        if row["expected_relation"] == "invariant":
            left = a["targets"] if q == "hue" else {q: a["targets"][q]}
            right = z["targets"] if q == "hue" else {q: z["targets"][q]}
            if left != right:
                raise ValueError("An invariance control changes its gold target")
        elif not a["targets"][q] < z["targets"][q]:
            raise ValueError("Increasing contrast not ordered")
        if row["kind"] == "factorial_one_axis_increase":
            changed = [k for k in a["primitive_axes"] if a["specified_values"][k] != z["specified_values"][k]]
            if changed != [row["changed_axis"]]:
                raise ValueError("Factorial contrast changes more than one primitive factor")
        if row["kind"] == "binding_increase":
            if Counter(re.findall(r"\w+", a["text"].lower())) != Counter(re.findall(r"\w+", z["text"].lower())):
                raise ValueError("Binding contrast has different word inventories")
    seen_comparisons = set()
    answer_groups = defaultdict(list)
    for c in b.comparisons:
        if c["comparison_id"] in seen_comparisons:
            raise ValueError("Duplicate comparison ID")
        seen_comparisons.add(c["comparison_id"])
        if c["candidate_meanings"][c["correct_index"]] != c["expected_relation"]:
            raise ValueError("Answer mapping is wrong")
        truth = "higher" if ((c["level_index"] > 2) != c["reversed_comparison"]) else "lower"
        if truth != c["expected_relation"]:
            raise ValueError("Reversed comparison has wrong ground truth")
        key = (c["quality_id"], c["entity_group_id"], c["template_id"], c["level_index"], c["reversed_comparison"])
        answer_groups[key].append(c)
    for questions in answer_groups.values():
        natural = [c for c in questions if c["format"] == "natural_continuation"]
        ab = [c for c in questions if c["format"] == "AB"]
        cells = {(c["answer_mapping"], c["option_order"]) for c in ab}
        if len(natural) != 1 or len(ab) != 4 or len(cells) != 4 or Counter(c["correct_index"] for c in ab) != {0: 2, 1: 2}:
            raise ValueError("Incomplete independent answer-label/printed-order counterbalancing")
    for t in b.triplets:
        a, near, far = [b.by_id[t[k]]["canonical_hue_degrees"] for k in ("anchor_case_id", "near_case_id", "far_case_id")]
        distance = lambda x, y: min(abs(x - y), 360 - abs(x - y))
        if not distance(a, near) < distance(a, far):
            raise ValueError("Invalid circular distance triplet")
    for domain, spec in b.factorial_domains.items():
        rows = [r for r in b.cases if r.get("factorial_domain") == domain and r["combination_split"] == "train"]
        for j, n in enumerate(spec["grid_shape"]):
            if {r["factorial_indices"][j] for r in rows} != set(range(n)):
                raise ValueError("Combination holdout removes an entire factor level")
    return {"case_types": dict(sorted(Counter(r["case_type"] for r in b.cases).items())),
            "cases": len(b.cases), "contrasts": len(b.contrasts), "behavior_questions": len(b.comparisons),
            "hue_triplets": len(b.triplets), "lexical_concepts": len(b.concepts),
            "lexical_labels": len(b.labels), "synthetic_entity_groups": len(b.entities),
            "qualities": len(QUALITIES) + 1, "factorial_domains": len(b.factorial_domains),
            "validation": "all structural and experimental-integrity checks passed"}


def make_profiles(b):
    by_split = {s: sorted(e["entity_id"] for e in b.entities if e["entity_split"] == s) for s in ("train", "validation", "test")}
    small = set(by_split["train"][:2] + by_split["validation"][:1] + by_split["test"][:1])
    medium = set(by_split["train"][:6] + by_split["validation"][:2] + by_split["test"][:2])
    profiles = {}
    for name in ("smoke", "laptop", "extended"):
        selected = []
        for r in b.cases:
            kind, group = r["case_type"], r["entity_group_id"]
            family = r["template_id"]
            q = r["quality_ids"][0]
            if name == "extended":
                keep = True
            elif name == "smoke":
                keep = group in small and (
                    (kind == "graded" and q in ("size", "mass", "temperature") and family in ("plain", "paraphrase") and r["level_index"] in (0, 2, 4)) or
                    (kind == "binding" and q == "size" and family in ("plain", "paraphrase")) or
                    (kind == "hue" and family == "numeric") or
                    (kind == "factorial" and r["factorial_domain"] == "extent_mass"))
            else:
                keep = (
                    kind == "lexical" or
                    (kind == "graded" and group in medium and (family in ("plain", "paraphrase") or (family == "numeric" and BY_KEY[q].numeric_sweep))) or
                    (kind == "controls" and group in medium and family in ("plain", "paraphrase") and r["condition"] != "negated_foil") or
                    (kind == "binding" and group in small) or
                    (kind == "numeric" and group in small) or
                    (kind == "hue" and group in medium and family in ("plain", "numeric")) or
                    (kind == "factorial" and group in small))
            if keep:
                selected.append(r["case_id"])
        ids = set(selected)
        questions = []
        for c in b.comparisons:
            if name == "extended":
                keep = True
            elif name == "smoke":
                keep = c["entity_group_id"] in small and c["quality_id"] in ("size", "mass", "temperature") and c["format"] == "natural_continuation" and c["template_id"] == "plain"
            else:
                keep = (c["entity_group_id"] in medium and c["format"] == "natural_continuation" and c["template_id"] in ("plain", "paraphrase")) or (
                    c["entity_group_id"] in small and c["quality_id"] in ("size", "mass", "temperature", "pitch", "pleasantness", "danger") and c["format"] == "AB" and c["template_id"] in ("plain", "paraphrase"))
            if keep:
                questions.append(c["comparison_id"])
        profiles[name] = dict(name=name, case_ids=selected, comparison_ids=questions,
            contrast_ids=[c["contrast_id"] for c in b.contrasts if c["low_case_id"] in ids and c["high_case_id"] in ids],
            triplet_ids=[t["triplet_id"] for t in b.triplets if {t["anchor_case_id"], t["near_case_id"], t["far_case_id"]} <= ids],
            case_count=len(selected), comparison_count=len(questions),
            planned_readouts_per_case=2,
            planned_extraction_vectors_per_layer_before_text_deduplication=len(selected) * 2,
            activation_bytes_at_2048_float32_per_layer=len(selected) * 2 * 2048 * 4,
            note="Selection only. Fit/evaluation filters still apply; do not fit all selected rows.")
    return profiles


PROTOCOL = r"""# Quality suite v3: design and interpretation

This is a reusable bank of ORIGINAL EXAMPLE DATA, not human norms, measurements,
model outputs, or evidence that every named attribute is a Gärdenforsian quality.
It extends the pilot while keeping typical knowledge, described properties,
referent binding, and causal use as separate questions.

## Labels and units

- `lexical`: author-proposed ordinary rankings, labeled `demo_ordinal`. All are
  provisional; inspect ambiguous, state-dependent, metaphorical and explicitly
  adjective-bearing entries. Do not cite them as independent ground truth.
- `graded`, `controls`, `binding`: the scene specifies an ordered state. Targets
  are DESIGN CODES 0, .25, .5, .75, 1. They do not mean equal psychological gaps.
  Word descriptions do not specify exact centimetres, temperatures, or masses.
  Only a numeric statement populates `specified_values` with physical values.
- `numeric`: the target key ends with `__value`, in the canonical physical unit.
  These are fictional stated measurements, not observations. Keep these targets
  separate from the normalized ordinal design codes. Unit-conversion statements
  may be rounded to eight significant digits; interpret equality to that precision.
- `factorial`: a complete grid of assigned attributes. Normalized design targets
  coexist with explicit physical values. The rectangular-block volume is derived
  from length, width and height, not an additional independent quality dimension.
- `hue`: standard HSV convention, not a perceptually uniform human color space.
  Predict sine/cosine or use circular distance. Two Cartesian output coordinates
  describe a one-dimensional circle; they do not establish two independent qualities.
  Color saturation/value are kept positive to avoid undefined hue at gray/black.
- Brightness versus luminance, loudness versus SPL, and elastic recovery versus
  elastic modulus are different constructs. The operational definitions in
  `qualities.json` matter. Affective/evaluative attributes specify a fictional
  observer or situation; they are not objective ratings or actual participant data.

Synthetic text with declared numbers tests whether a model represents and binds
stated attributes. It does not by itself show spontaneous world knowledge,
perceptual grounding, or a privileged semantic geometry. Report word-only and
numeric conditions separately, as well as transfer between them.

Fit inexpensive baselines in the same training folds: character/token counts,
unigram TF-IDF, averaged static input embeddings, and numeric-literal features for
numeric conditions. Explicit adjectives and numbers make many ordinary rows easy.
The same-word-inventory binding pairs are essential: a bag-of-words predictor
cannot distinguish their changed referent assignments. Compare context-free lexical
knowledge with context-bound tracking, rather than treating either as the other.

## Experimental families

1. Typical lexical associations, with noun-length and category controls.
2. Five ordered states of the same fictional item, across lexical/numeric forms.
3. Irrelevant additions, changes to another item, and explicitly denied foil claims.
4. Same-word-inventory role swaps, both clause orders and both queried entities.
5. Numeric interpolation, bounded range tests, selected extrapolation, equivalent units.
6. Independent factorial attributes: extent/mass; length/width/height;
   roughness/hardness/recovery; pitch/SPL; pleasantness/arousal/danger;
   hue/saturation/value; speed/temperature.
7. Circular hue neighborhoods and the equivalence of 0 and 360 degrees.
8. Natural-completion behavioral questions plus independent A/B meaning and
   printed-order counterbalancing. Reversing the queried pair reverses the truth.

## Splits and independent units

All variants of a synthetic entity family remain in one entity split, including
its companion item. The identities are invented names: held-out synthetic names
are not equivalent to held-out real-world object categories or people.
Plain/numeric wording is available for fitting; paraphrase/reordered wording is
reserved. Validate settings on validation ENTITIES using training wording first.
For fitting factorial models, also require `combination_split == 'train'`.
Report excluded entities, excluded wording, and excluded combinations separately
and jointly. None is a random prompt-row split.

Numeric sweep files are challenge-only; do not include them in fitting because
their template happens to be `numeric`. Fit a numeric predictor on stated values
in training graded/factorial rows, not on their design-code labels, before evaluating
the `__value` challenge. Declare any log/unit transform and fit calibration on train.

Lexical groups merge identical normalized spellings only. Review synonyms and
near-duplicates before empirical work. Exact names detected in the old pilot's
concept CSV are assigned to training and marked as overlaps; this does not detect
all semantic overlap. These example holdouts are exploratory, not independently
collected confirmation. Template and entity files are transparent, not concealed.

Bootstrap complete entity/concept families, with their variants, and acknowledge
the much smaller count of independent constructed scenarios than prompt rows.
Do not claim thousands of natural observations from thousands of generated sentences.
Use independently collected, appropriately licensed norms or new ratings later.
The empty human-rating template contains no invented participants or ratings.

## Wider search: axes, subspaces, and local behavior

Start with one direction per quality, then examine the span of MATCHED training
contrast vectors. Use SVD, rank curves, bootstrap subspace stability and held-out
semantic performance to test candidate subspace ranks [1,2,3,4,6,8,12,16]. Limit PCA
or contrast-SVD rank by sample/feature rank. Reduced-rank regression additionally
has an output-rank limit; many settings are inapplicable in small domains.

PCA can be solved with SVD; ridge has a convex quadratic objective. Repeated starts
do not discover additional meaningful local optima for these estimators. With a
single scalar target, PCA followed by linear regression is still one overall linear
functional of the original activation. More retained PCs do not prove that a
quality has that many intrinsic dimensions.

For several components, use multi-output prediction on the factorial grids,
reduced-rank regression or a shared low-dimensional subspace with separate heads.
Compare a shared subspace against property-specific directions and cross-domain
transfer. Use raw or train-centered residuals; document any altered metric.
Compare subspaces by principal angles/projectors, not arbitrary rotated columns.

To investigate LOCAL structure, fit prespecified adjacent-range contrasts (0→1,
1→2, 2→3, 3→4) on training entities, and compare their alignment and transfer.
Also compare domains and wording families. This asks whether a global axis bends
or becomes context-dependent; it is distinct from an optimizer having local minima.
With only five qualitative levels, do not claim a finely resolved manifold.
Use the numeric sweeps as explicit, separate interpolation/range diagnostics.

Only after linear baselines, optionally try a small signed dictionary-learning
model or a small nonlinear probe on cached features. Use 5 random initializations
for exploration, up to 10 for a specified replication; save all outcomes, convergence
and held-out validation scores. Do not choose a restart using test labels. Lower
reconstruction loss alone does not establish semantic interpretability. NMF assumes
nonnegative inputs and is not a default method for signed residual activations.

Use grouped nested validation for ranks, layers, regularization and nonlinear
choices. Start at blocks [4,8,12,16,20,24,28] for the current 28-block model, then
refine nearby layers on development data only. Broad searching increases selection
opportunities; log all tried settings and reserve a new final confirmatory sample.
Prefer interval estimates and held-out predictive improvement. Any formal screen
across many qualities/layers must define its testing family and correction.

## Causal follow-up

First verify behavioral competence for EACH property/format using development
data. Binary A/B words and natural continuations have different biases; score both
candidate continuations exactly, including multi-token answers, with validated
token boundaries. No answers appear after the prefix in these files.

Test an entity position after all context and the final prompt position separately.
Preserve pre-final-normalization block hook semantics. An earlier-token intervention
after the last attention block cannot propagate to a later answer position.

After competence and numerical-resolution checks, test matched-state patches and
projection replacement before enlarging additive doses. For multiple qualities,
measure a cross-property intervention matrix: intervention on quality j versus
predictions/judgments for quality k, with other factorial attributes held fixed.
Check changes against baseline variability rather than assuming raw scales match.
Keep identity/tag controls, matched-norm random directions, A-bias decomposition,
and precision checks. Prediction, described-state tracking and selective causal
influence remain separate outcomes.

## Primary references

- Grand et al., semantic projection and human feature judgments:
  https://www.nature.com/articles/s41562-022-01316-8
  The article links collected data at https://osf.io/5r2sz/ . These examples DO NOT
  reproduce or substitute for those collected data; category rating scales may differ.
- Park et al., linear representation geometry:
  https://proceedings.mlr.press/v235/park24c.html
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- NMF: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- Activation patching: https://arxiv.org/abs/2404.15255
"""

CODEX_PROMPT = r"""# Integrate the quality suite into the existing project

Read `docs/QUALITY_SUITE_V3.md`, `data/quality_suite_v3/README.md`,
`data/quality_suite_v3/manifest.json`, and `configs/quality_suite_v3.json`.
Implement support in the existing repository; preserve original pilot runs/caches.
This JSON is a SUITE SPECIFICATION, not a claim that the old size-only CLI already
accepts it. Inspect the real loaders, schemas and tests before changing them.

1. Add a versioned suite loader and a separate `qd suite ...` command family (or
   equally explicit compatible subcommands). Do not map every attribute into
   `size_score`, overwrite `data/concepts.csv`, or silently apply the old size prompt.
   Validate IDs, foreign keys, source kinds, group splits, prompt template roles and
   held-out combination cells. Reuse cache entries only when actual prompt tokens,
   model revision, dtype and hook/readout semantics agree.
2. Load selected case/question IDs from the named profile. `smoke` verifies code,
   `laptop` is the initial broad screen, and `extended` is optional. First print
   forward counts, independent group counts and expected activation storage.
   Use the same Qwen3-1.7B-Base, MPS FP16, batch 1, selective positions and resumable
   60-minute budget. No automatic multi-model run or exhaustive steering sweep.
3. Extract the complete-block residual after context, before final stack RMSNorm.
   Convert character readout spans to token spans using offsets from the pinned
   tokenizer; use the last overlapping entity subtoken and separately the final
   prompt token. Assert the spans are valid; the text files cannot provide token
   offsets before tokenization. Never pool an initial pre-context mention into
   the contextual readout. Capture one underlying activation per identical text/
   position/layer, attaching several property labels afterward.
4. Use training entities + training templates; for factorial fitting also require
   training combinations. Numeric challenge rows cannot enter fitting. Tune on
   validation entities, then report each held-out axis of generalization. Avoid
   random prompt-row splits. Preserve ambiguity about synonym grouping in lexical
   seeds and keep their claims separate from controlled synthetic cases.
5. Fit per-property mean differences, ridge, and train-only contrast SVD subspaces.
   For several independent outputs, implement multi-output/reduced-rank models.
   Respect sample/feature rank for PCA/contrast SVD, and also output rank for
   reduced-rank regression. Compare with token-length, unigram, static-embedding
   and numeric-literal baselines, as applicable. Fit preprocessing and regularization
   within grouped training folds.
   PCA-to-ridge for one scalar remains one linear readout; do not call its PC count
   the intrinsic dimension of the quality. Treat hue with sine/cosine outputs and
   circular error; never apply an ordinary increasing-hue Spearman criterion.
6. Compare the adjacent-level directions and their subspaces across ranges,
   templates and domains. Generate paired contrast, binding, control, interpolation,
   unit-invariance, circular-neighborhood and factorial-selectivity results.
   The `targets` design codes are ordinal. Use `specified_values` for physical-unit
   diagnostics, and do not mix numeric `__value` targets with ordinal codes.
   Volume is derived, and HSV value is not emitted luminance. Note the different
   scales and sample counts instead of pooling every score into one headline.
7. Add optional nonlinear/dictionary exploration on CPU-cached data only, disabled
   by default. Distinguish optimizer restarts from local regions of the representation.
   Log every restart; use validation selection, early stopping and subspace stability.
   Require a documented held-out improvement over linear baselines before escalating.
8. Evaluate per-property behavioral competence and answer bias before causal work.
   Some natural answers contain multiple tokens. Implement exact teacher-forced
   conditional sequence scoring and verify it, or mark that format unsupported;
   never use only the first token. Distinguish answer meaning from printed A/B order.
   Give matched baseline/steering comparisons on identical subjects and save raw
   scores. Verify float32 final-head log-odds on a small sample where FP16 effects
   are tiny. Keep interventions at validated sites with subsequent computational
   paths to the measured output.
9. Run targeted interventions only for a small development-selected set of
   properties with competent behavioral formats. Test the cross-property effect
   matrix using factorial one-axis changes, ordinary random controls and identity
   controls. Preserve negative findings and the v2 paired-analysis corrections.
10. Add focused tests for scientific invariants, load/filter joins, hue wrapping,
    multi-output targets, factorial isolation, train-only search, sequence scoring
    and cache reuse. Execute a real smoke run where available and then a bounded
    laptop screen. Do not require a positive scientific result in any test.
11. Produce raw CSV/JSON results, readable plots, an honest report with original
    versus v3 comparisons, label provenance, independent counts, tested hypotheses,
    all model-selection choices, failures and limits. A thousand constructed rows
    are not a thousand independent human observations. Add an importer for actual
    independent labels without inventing missing ratings.

Proceed with implementation and verification. Report exact commands and what
actually ran. Do not stop after another plan, fabricate model results, or claim
that the data-population script has already implemented this integration.
"""


def target_registry():
    registry = {}
    for q in QUALITIES:
        registry[q.key] = dict(kind="constructed_ordinal_design_code", levels=[0, .25, .5, .75, 1],
                               physical_unit=q.unit, physical_values=q.values,
                               note="Only specified_values in numeric text are physical quantities; ordinal gaps are not psychological distances.")
        if q.numeric_sweep:
            registry[q.key + "__value"] = dict(kind="stated_numeric_challenge", unit=q.unit, source_quality=q.key)
    for key in ("length", "width", "height", "saturation", "value"):
        registry[key] = dict(kind="constructed_factorial_code", levels=[0, .5, 1],
                             physical_unit="cm" if key in ("length", "width", "height") else "percent")
    registry["hue_sin"] = dict(kind="circular_coordinate", geometry="HSV hue", coupled_with="hue_cos")
    registry["hue_cos"] = dict(kind="circular_coordinate", geometry="HSV hue", coupled_with="hue_sin")
    return registry


def bundle(builder, legacy_fields):
    b = builder
    summary = validate(b)
    profiles = make_profiles(b)
    files = {}
    for kind in sorted(summary["case_types"]):
        files[f"{BASE}/cases/{kind}.jsonl"] = "".join(dumps(r) + "\n" for r in b.cases if r["case_type"] == kind)
    files[f"{BASE}/comparisons.jsonl"] = "".join(dumps(r) + "\n" for r in b.comparisons)
    files[f"{BASE}/contrasts.csv"] = csv_text(b.contrasts, list(b.contrasts[0]))
    files[f"{BASE}/hue_triplets.csv"] = csv_text(b.triplets, list(b.triplets[0]))
    files[f"{BASE}/concepts.csv"] = csv_text(b.concepts.values(), ["concept_id", "lemma", "synonym_group", "category", "source_kind", "entity_split", "previous_pilot_overlap"])
    files[f"{BASE}/quality_labels.csv"] = csv_text(b.labels, ["concept_id", "quality_id", "ordinal_score", "label_kind", "label_source", "notes"])
    files[f"{BASE}/entities.csv"] = csv_text(b.entities, ["entity_id", "name", "foil", "entity_split", "tag"])
    files[f"{BASE}/human_ratings_template.csv"] = csv_text(
        [{"concept_id": r["concept_id"], "quality_id": r["quality_id"]} for r in b.labels],
        ["concept_id", "quality_id", "rater_id", "rating", "scale_id", "source", "notes"])
    qualities = [asdict(q) for q in QUALITIES] + [dict(key="hue", domain="visual", geometry="circular",
        unit="HSV degrees", hue_labels=HUES, note="Convention-defined synthetic hue; not a perceptually uniform color metric.")]
    files[f"{BASE}/qualities.json"] = pretty(qualities)
    files[f"{BASE}/target_registry.json"] = pretty(target_registry())
    files[f"{BASE}/factorial_domains.json"] = pretty(b.factorial_domains)
    for name, profile in profiles.items():
        files[f"{BASE}/profiles/{name}.json"] = pretty(profile)
    config = dict(schema_version=VERSION, suite=SUITE, configuration_kind="suite_spec_not_legacy_qd_config",
        dataset_root=BASE, default_profile="laptop", model_id="Qwen/Qwen3-1.7B-Base", model_revision="reuse_and_record_existing_pinned_revision",
        backend="mps", dtype="float16", batch_size=1, max_tokens=256, max_model_minutes=60,
        extraction_block_numbers=[4, 8, 12, 16, 20, 24, 28], readouts=["last_contextualized_entity_subtoken", "final_prompt_token"],
        subspace_ranks=[1, 2, 3, 4, 6, 8, 12, 16], ridge_alphas=[.1, 1, 10, 100, 1000],
        fit_filter=dict(entity_split="train", template_split="train", combination_split="train", exclude_case_types=["numeric"]),
        nonlinear=dict(enabled=False, restart_seeds=[11, 23, 37, 53, 71], maximum_restarts=10,
                       selection="validation_only", require_grouped_validation=True),
        intervention=dict(enabled=False, maximum_initial_properties=3, strengths=[-1, -.5, 0, .5, 1],
                          gate="per_property_behavioral_competence_and_numerical_checks"),
        interpretation="Examples and engineering controls; independently labeled confirmation remains separate.")
    files["configs/quality_suite_v3.json"] = pretty(config)
    files["docs/QUALITY_SUITE_V3.md"] = PROTOCOL
    files["docs/CODEX_QUALITY_SUITE_V3.md"] = CODEX_PROMPT
    table = "\n".join(f"| {k} | {v:,} |" for k, v in summary["case_types"].items())
    ptable = "\n".join(f"| {n} | {p['case_count']:,} | {p['comparison_count']:,} |" for n, p in profiles.items())
    files[f"{BASE}/README.md"] = f"""# Quality suite v3 example data

Generated by `populate_quality_suite.py` {VERSION}; seed {b.seed}. No inference or downloads occur.
Existing top-level data, Python package, old configurations, runs and caches are not changed.

**15 candidate qualities, {len(b.entities)} synthetic entity families, {summary['lexical_concepts']} lexical concepts, and 7 factorial domains.**
Labels are author-proposed examples or explicitly assigned fictional values, not collected human data.

| Case family | Rows |
|---|---:|
{table}

There are also {summary['contrasts']:,} paired contrasts, {summary['behavior_questions']:,} behavioral questions,
and {summary['hue_triplets']:,} circular similarity triplets. These are repeated constructed scenarios,
not that many independent observations. Data generation is cheap; do not automatically infer every row.

| Profile | Activation cases | Behavioral questions |
|---|---:|---:|
{ptable}

## Use in the existing project

The current pilot code was built for size-only data. This generator adds separate files;
it does NOT claim the old `qd run` accepts the new schema. Give Codex
`docs/CODEX_QUALITY_SUITE_V3.md` to implement the adapter and broader analyses.
Do not rename every quality to size or overwrite `data/concepts.csv`.

Read `docs/QUALITY_SUITE_V3.md` for the full experiment and search protocol.
`configs/quality_suite_v3.json` is a suite specification for that integration.

## File schema

- `cases/*.jsonl`: complete text, entity/template/combination splits, target dictionaries,
  declared numeric values where present, provenance, and a CHARACTER span for the final
  contextualized entity. Token positions require the actual pinned tokenizer.
- `quality_labels.csv`: provisional lexical rankings, with source labels.
- `concepts.csv`: global normalized-spelling groups. Automatic semantic synonym resolution
  is not claimed. Manually check synonyms and near-duplicates before final inference.
- `contrasts.csv`: source/target case IDs and the single changed axis or invariance condition.
- `comparisons.jsonl`: prefixes WITHOUT supplied answers, candidate continuations,
  semantic meanings, correct indices, and independent label/printed-order counterbalancing.
  Natural candidates can be MULTIPLE TOKENS; score full conditional sequences correctly.
- `hue_triplets.csv`: circular near/far relationships, not ordinary linear hue order.
- `target_registry.json`: design codes versus physical values versus circular outputs.
- `profiles/*.json`: exact selected IDs; selection is not permission to fit held-out rows.
- `human_ratings_template.csv`: empty rating fields only; no fake participants.

## Running the generator

From the project root, after saving the script there:

```bash
python3 populate_quality_suite.py --dry-run
python3 populate_quality_suite.py
python3 populate_quality_suite.py --verify
```

An identical rerun leaves unchanged files untouched. Differing generated files cause a
preflight refusal before any writes. `--force` explicitly replaces only this suite's
listed output files. It never edits the original pilot files. Keep hand edits elsewhere
or back them up before requesting replacement. `--entities 10` creates a smaller bank;
the default is 20. A changed seed or entity count is a different dataset specification.

`python3 populate_quality_suite.py --self-test` runs generation, invariance, leakage,
factorial isolation, circular, counterbalancing and safe-write tests in temporary directories.

The legacy concepts CSV header observed during generation was:
`{', '.join(legacy_fields) if legacy_fields else '(not present or no recognized header)'}`.
Only exact recognized lemma/name/noun/word overlaps with that file are detected.
No existing source-code schema or loader compatibility has been inferred from this header.

## Minimal inspection (standard library only)

```python
import json
from pathlib import Path
base = Path("data/quality_suite_v3")
selected = set(json.loads((base / "profiles/smoke.json").read_text())["case_ids"])
for path in sorted((base / "cases").glob("*.jsonl")):
    for line in path.open():
        row = json.loads(line)
        if row["case_id"] in selected:
            print(row["case_id"], row["targets"], row["text"])
```

The generator validates its own examples. It has not run or validated LLM findings.
"""
    files[f"{BASE}/manifest.json"] = pretty(dict(schema_version=VERSION, suite=SUITE, seed=b.seed,
        entity_count=len(b.entities), summary=summary, profiles={n: {k: p[k] for k in ("case_count", "comparison_count")} for n, p in profiles.items()},
        existing_concepts_header=legacy_fields, previous_pilot_exact_lemmas=sorted(b.previous),
        split_unit="entity_family_or_exact_normalized_lexical_form", source=AUTHOR,
        generated_files={path: digest(content) for path, content in sorted(files.items())}))
    return files, summary, profiles


def safe_target(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Unsafe generated relative path")
    target = root / rel
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Refusing symlink/outside-root write: {relative}")
    return target


def install(root, files, force=False, dry_run=False):
    # Preflight the complete batch; never half-update merely because a later file conflicts.
    targets = {p: safe_target(root, p) for p in files}
    conflicts = [p for p, target in targets.items() if target.exists() and
                 (not target.is_file() or target.read_bytes() != files[p].encode("utf-8"))]
    if conflicts and not force:
        raise ValueError("Different files already exist; nothing written. Review changes before --force:\n" + "\n".join(conflicts[:12]))
    if any(target.exists() and not target.is_file() for target in targets.values()):
        raise ValueError("A generated file path is occupied by a directory; nothing written")
    if dry_run:
        return 0
    written = 0
    for p, target in targets.items():
        raw = files[p].encode("utf-8")
        if target.is_file() and target.read_bytes() == raw:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".quality-suite-", delete=False) as f:
                tmp_name = f.name
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, target)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        written += 1
    return written


def verify_installed(root):
    path = safe_target(root, f"{BASE}/manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("suite") != SUITE:
        raise ValueError("Wrong suite manifest")
    errors = []
    allowed_special = {"configs/quality_suite_v3.json", "docs/QUALITY_SUITE_V3.md", "docs/CODEX_QUALITY_SUITE_V3.md"}
    for relative, expected in manifest["generated_files"].items():
        if not (relative.startswith(BASE + "/") or relative in allowed_special):
            raise ValueError("Manifest contains a path outside this suite's ownership")
        p = safe_target(root, relative)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    if errors:
        raise ValueError("Missing or changed generated files:\n" + "\n".join(errors))
    return manifest


def self_test():
    for q in QUALITIES:
        assert len(q.values) == len(q.words) == len(q.alternate) == len(q.anchors) == 5
        assert all(len(g) == 5 for g in q.anchors), q.key
        assert list(q.values) == sorted(set(q.values)), q.key
    b = Builder(entity_count=10, previous={"rabbit"}).build()
    files, summary, profiles = bundle(b, ["concept_id", "lemma", "size_score"])
    assert all(r["entity_split"] == "train" for r in b.cases if r.get("prior_pilot_overlap"))
    assert hue_targets(0) == hue_targets(360)
    assert set(profiles["smoke"]["case_ids"]) <= set(profiles["extended"]["case_ids"])
    assert set(profiles["laptop"]["case_ids"]) <= set(profiles["extended"]["case_ids"])
    # Known mathematical conversions, including the negative-temperature boundary.
    assert math.isclose(((-8 * 9 / 5 + 32) - 32) * 5 / 9, -8)
    assert math.isclose(1000 / 1000, 1)
    assert math.isclose(math.prod([5, 15, 45]), 3375)
    # Deliberately corrupted experimental controls must be detected.
    invariant = next(c for c in b.contrasts if c["kind"] == "irrelevant_invariant")
    altered = b.by_id[invariant["high_case_id"]]
    old = dict(altered["targets"])
    altered["targets"][invariant["quality_id"]] = .9
    try:
        try:
            validate(b)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed to detect incorrect invariance labels")
    finally:
        altered["targets"] = old
    with tempfile.TemporaryDirectory(prefix="quality_suite_test_") as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        sentinel = root / "data/concepts.csv"
        sentinel.write_text("concept_id,lemma,size_score\nold,rabbit,3\n")
        original = sentinel.read_bytes()
        assert install(root, files, dry_run=True) == 0
        assert not (root / BASE).exists()
        assert install(root, files) == len(files)
        before = (root / BASE / "cases/graded.jsonl").stat().st_mtime_ns
        assert install(root, files) == 0
        assert (root / BASE / "cases/graded.jsonl").stat().st_mtime_ns == before
        verify_installed(root)
        edited = root / BASE / "README.md"
        edited.write_text("user edit\n")
        additional = dict(files)
        additional[f"{BASE}/should_not_be_written.txt"] = "must not be written"
        try:
            install(root, additional)
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting files were not protected")
        assert not (root / BASE / "should_not_be_written.txt").exists()
        assert edited.read_text() == "user edit\n"
        install(root, files, force=True)
        verify_installed(root)
        assert sentinel.read_bytes() == original
        try:
            install(root, {"../outside.txt": "bad"})
        except ValueError:
            pass
        else:
            raise AssertionError("Path traversal accepted")
        # Saved bytes are independently parsed, not merely checked in memory.
        parsed = [json.loads(line) for p in (root / BASE / "cases").glob("*.jsonl") for line in p.open(encoding="utf-8")]
        assert len(parsed) == summary["cases"]
    # Same construction parameters must reproduce exactly the same file bytes.
    again, _, _ = bundle(Builder(entity_count=10, previous={"rabbit"}).build(), ["concept_id", "lemma", "size_score"])
    assert files == again
    print("PASS: schemas, counterbalancing, matched controls, entity splits, factorial isolation,")
    print("circular geometry, deterministic generation, safe reruns, conflict preflight and original-file preservation.")
    print(pretty(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Existing repository root; default current directory")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--entities", type=int, choices=(10, 20), default=20, help="Invented entity families; default 20")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory; write nothing")
    parser.add_argument("--force", action="store_true", help="Explicitly replace differing generated suite files only")
    parser.add_argument("--verify", action="store_true", help="Check the installed suite's saved file hashes")
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests in temporary directories")
    parser.add_argument("--list-qualities", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.list_qualities:
        for q in QUALITIES:
            print(f"{q.key:14s} {q.domain:12s} {q.note}")
        print("hue            visual       Circular HSV example; use sine/cosine and circular distance.")
        return 0
    root = args.project_root.expanduser().resolve()
    if not root.is_dir() or not (root / "data").is_dir() or not ((root / "pyproject.toml").is_file() or (root / "src/quality_dimensions").is_dir()):
        parser.error("Use the existing project root (with data/ and pyproject.toml or src/quality_dimensions). Pass --project-root explicitly if needed.")
    if args.verify:
        manifest = verify_installed(root)
        print(f"Verified {len(manifest['generated_files'])} generated files. No model results were checked.")
        return 0
    fields, previous = legacy_info(root)
    b = Builder(args.seed, args.entities, previous).build()
    files, summary, profiles = bundle(b, fields)
    count = install(root, files, args.force, args.dry_run)
    print(pretty(summary))
    print(f"{'DRY RUN: would manage' if args.dry_run else 'Suite contains'} {len(files)} files; {sum(len(s.encode('utf-8')) for s in files.values()) / 1048576:.1f} MiB.")
    if not args.dry_run:
        print(f"Wrote {count} new/changed files to {root}; unchanged files were left intact.")
    for name, p in profiles.items():
        print(f"{name}: {p['case_count']:,} activation cases; {p['comparison_count']:,} behavioral questions.")
    print("Next: give docs/CODEX_QUALITY_SUITE_V3.md to Codex to integrate the new loader and analyses.")
    print("The existing size-only qd run command has not been modified. No inference was run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
