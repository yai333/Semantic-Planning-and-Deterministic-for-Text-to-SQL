from __future__ import annotations

import re as _re_mod

from collections import Counter
from typing import Any, Mapping, Sequence

ROUTE_DETAIL_CAP = 40

RESOLVERS = frozenset({
    "list_subjects", "find_metric", "find_attribute", "search_values", "find_paths",
})

def _searchable(skills: Any, concept: str) -> tuple[str, ...]:
    try:
        target = skills.ontology.concept(concept)
    except Exception:                       # noqa: BLE001
        return ()
    return tuple(a.name for a in getattr(target, "searchable_attributes", ()))

def _prefer_display(skills: Any, concept: str, qualified: str) -> str:
    """A phrase that lands on the KEY resolves to the display instead.

    `display` is the ontology stating which attribute identifies a concept TO A
    READER; `key` is what rows join on. Where they differ the key is an
    implementation detail no answer should project, and every ACME gold agrees:
    a policy is identified by `policy_number`, a claim by
    `company_claim_number`, never by the `*_Identifier` column.

    Nothing changes for a concept whose display IS its key -- PolicyHolder and
    Agent are identified by the party identifier, and `display` says so -- so
    this cannot rewrite the role questions that already pass.
    """
    if "." not in qualified:
        return qualified
    owner, _, name = qualified.rpartition(".")
    try:
        target = skills.ontology.concept(concept)

        keys = {str(k) for k in (getattr(target, "key", ()) or ())}
        column = str(getattr(target.attribute(name), "column", "") or "")
        display = getattr(getattr(target, "title_attribute", None), "name", None)
    except Exception:                           # noqa: BLE001
        return qualified
    if display and display != name and (name in keys or (column and column in keys)):
        return f"{owner}.{display}"
    return qualified

def _date_attributes(skills: Any, concept: str) -> tuple[str, ...]:
    """The concept's DATE-typed attributes, per the ontology's `value_type`.

    The counterpart of `_searchable` for time. A string literal is grounded by
    asking the warehouse how the value is spelled; a period is grounded by
    calendar arithmetic -- but WHICH attribute it bounds is an ontology fact
    either way, and this is where that fact is read rather than assumed.
    """
    try:
        target = skills.ontology.concept(concept)
    except Exception:                       # noqa: BLE001
        return ()
    return tuple(a.name for a in getattr(target, "attributes", {}).values()
                 if str(getattr(a, "value_type", "")).casefold() == "date"
                 or str(getattr(a, "type", "")).casefold() == "date")

_DETERMINERS = frozenset({"each", "every", "all", "the", "a", "an", "any",
                          "our", "their", "its", "his", "her", "these",
                          "those", "this", "that"})

_AVERAGE_LEAD = _re_mod.compile(r"^(?:the )?average\b|\bavg\b",
                                _re_mod.IGNORECASE)

_ENUMERATION_LEAD = _re_mod.compile(
    r"^(?:all|every(?:\s+one)?\s+of|each\s+of)\b", _re_mod.IGNORECASE)

_AGGREGATE_SIGNAL = _re_mod.compile(
    r"\b(?:total|sum|summation|average|avg|combined|overall|how\s+much|"
    r"how\s+many)\b", _re_mod.IGNORECASE)

_TIME_STOPWORDS = frozenset({
    "in", "on", "at", "of", "the", "a", "an", "during", "within", "that",
    "were", "was", "is", "are", "been", "be", "year", "yr",
})

_ANAPHORS = frozenset({"corresponding", "respective", "associated"})

def _deanaphored(phrase: str) -> str:
    """A span with anaphoric adjectives removed, for METRIC LOOKUP only.

    The span itself is reported verbatim everywhere; only what is sent to
    `find_metric` changes. Unlike `_undetermined` this applies to quantity
    spans too, because an anaphor -- unlike a determiner -- can never be part
    of a metric's own name or alias, so removing it cannot move a resolution
    between two metrics; it can only stop the connective from diluting the
    match.
    """
    words = [w for w in (phrase or "").split()
             if w.casefold().strip(",") not in _ANAPHORS]
    return " ".join(words) or phrase

def _undetermined(phrase: str) -> str:
    """A definitional span with its leading article removed, for LOOKUP only.

    The planner quotes verbatim and English varies: the same definition arrives
    as "number of policies" on one run and "the number of policies" on the next.
    Against `find_metric` those are not equivalent --

        'number of policies'      -> unique   PolicyCount
        'the number of policies'  -> partial

    -- so a fact the QUESTION states landed or did not depending on an article,
    which is why the definitional questions were FLAKY rather than broken.

    DEFINITIONAL SPANS ONLY. Applied to `quantity_phrases` as well, this measured
    WORSE overall: query-724db899 fell 3/3 -> 1/3 and query-244778fe 3/3 -> 0/3,
    both LISTING questions, where changing which metric a quantity phrase
    resolves to changes what gets projected. A definitional span is only ever
    read to decide what a term MEANS, so normalising it cannot move a
    projection. Narrow on purpose, and the wider version is recorded here
    because it looked more principled and was not.
    """
    words = [w for w in (phrase or "").split() if w]
    while words and words[0].casefold().strip(",") in _DETERMINERS:
        words.pop(0)
    return " ".join(words) or phrase

def _period(phrase: str) -> tuple[str, str] | None:
    """A span of time as inclusive ISO bounds, or None if it names no period.

    WHAT A YEAR MEANS IS CALENDAR ARITHMETIC, NOT ONTOLOGY AND NOT PROMPTING.
    It was briefly neither: SKILL.md spelled out an attribute name and a pair of
    2019 dates, which is the benchmark's own answer sitting in the prompt. It
    lifted one question from 4/5 to 4/4 and would have taught the planner
    nothing about any other. So the arithmetic lives here, in code, where it is
    the same for every question and can be read.

    Deliberately three forms and no more -- a year, a month, a day. Relative
    periods ("last quarter") need a clock and an as-of convention this project
    has not chosen, and guessing one would be inventing semantics on the
    warehouse's behalf. Anything else returns None and is REPORTED unresolved.
    """
    import calendar
    import re as _re

    text = phrase.strip()
    if m := _re.fullmatch(r".*?(\d{4})-(\d{2})-(\d{2}).*", text):
        day = "-".join(m.groups())
        return day, day
    if m := _re.fullmatch(r".*?(\d{4})-(\d{2}).*", text):
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            last = calendar.monthrange(year, month)[1]
            return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"
        return None
    years = _re.findall(r"\b(1[89]\d{2}|2[01]\d{2})\b", text)
    if len(years) == 1:
        return f"{years[0]}-01-01", f"{years[0]}-12-31"
    return None

def _summands(skills: Any, metric: str) -> tuple[str, ...]:
    """Every metric added into `metric`. Delegates to the ontology on purpose.

    This walk USED to live here, which meant the advice `review_plan` gives and
    the rejection `agent._certify` makes were two functions computing the same
    thing -- two chances to disagree about which plans are legal. It is now
    `Ontology.summands`, for the reason `measured_over` states in that file.
    """
    try:
        return tuple(skills.ontology.summands(metric))
    except Exception:                       # noqa: BLE001
        return ()

def _leaf_metrics(skills: Any, metric: str) -> tuple[str, ...]:
    """`metric` if it is a leaf, else every leaf beneath it.

    A composite has no operand and therefore no route of its own; what needs
    routing are the metrics it is built from.
    """
    onto = getattr(skills, "ontology", None)
    if onto is None:
        return (metric,)
    out: list[str] = []

    def walk(name: str, guard: frozenset[str]) -> None:
        if name in guard:
            return
        entry = onto.metrics.get(name)
        if entry is None:
            return
        if entry.is_composite:
            for component in entry.components:
                walk(component, guard | {name})
        elif name not in out:
            out.append(name)

    walk(metric, frozenset())
    return tuple(out) or (metric,)

def _why_no_subject(decomp: Mapping[str, Any], unknown: Sequence[str]) -> str:
    """The abstention's REASON, from the record. Two are not the same finding:
    naming nothing is a planner that could not read the menu; naming something
    ungoverned is one that read it and answered off it."""
    if unknown:
        return f"named {list(unknown)[:3]}, which the ontology does not govern"
    if not (decomp.get("subjects") or []):
        return "the decomposition named no subject at all"
    return "no named subject survived validation"

def resolve(decomposition: Mapping[str, Any], question: str, skills: Any) -> dict:
    """Every span through the deterministic resolvers. No model call.

    A span that resolves to nothing is REPORTED, never dropped: the planner is
    told what it asked for and did not get, which is the difference between a gap
    and a silent guess.
    """
    decomp = decomposition

    calls: Counter = Counter()

    def data(result):
        return result.data if hasattr(result, "data") else result

    def invoke(name: str, *args, **kw):

        if name not in RESOLVERS:
            raise LookupError(
                f"resolve.py invoked {name!r}, which it does not declare in "
                f"RESOLVERS ({sorted(RESOLVERS)}) -- declare it there and in "
                f"SKILL.md's `resolvers:`, or do not call it"
            )
        calls[name] += 1
        return data(getattr(skills, name)(*args, **kw))

    known = set(skills.ontology.concept_names())

    subjects = [s for s in (decomp.get("subjects") or []) if s in known]
    unknown = [s for s in (decomp.get("subjects") or []) if s not in known]

    def span(raw: Any) -> str:

        text = str(raw).strip()
        pairs = ('""', "''", "``", "“”", "‘’")
        changed = True
        while changed and len(text) > 1:
            changed = False
            for open_q, close_q in pairs:
                if text[0] == open_q and text[-1] == close_q:
                    text, changed = text[1:-1].strip(), True
                    break
        return text

    import re as _re

    haystack = question.casefold()
    off_question: list[dict] = []

    def occurs(text: str) -> bool:
        return bool(_re.search(rf"(?<!\w){_re.escape(text.casefold())}(?!\w)", haystack))

    SHORT = {"quantity_phrases": "quantity", "attribute_phrases": "attribute",
             "literal_phrases": "literal", "comparison_phrases": "comparison",
             "rank_phrases": "rank", "time_phrases": "time"}

    def spans(slot: str) -> list[str]:
        out = []
        for raw in decomp.get(slot) or []:
            text = span(raw)
            if not text:
                continue
            if not occurs(text):
                off_question.append({"slot": SHORT.get(slot, slot), "span": text[:60],
                                     "as_returned": str(raw)[:60]})
            out.append(text)
        return out

    report: dict[str, Any] = {"unknown_subjects": unknown,
                              "unresolved": [], "ambiguous": []}

    recovered: list[str] = []

    name_words = {
        name: _re.sub(r"(?<!^)(?=[A-Z])", " ", name).casefold()
        for name in known}

    def _name_pattern(words: str) -> str:

        stem = _re.escape(words)
        if words.endswith("y"):
            return f"{_re.escape(words[:-1])}(?:y|ies)"
        return f"{stem}(?:|s|es)"

    def _occurrences(words: str):
        return _re.finditer(rf"(?<!\w){_name_pattern(words)}(?!\w)", haystack)

    starts: dict[int, int] = {}
    for name, words in name_words.items():
        for m in _occurrences(words):
            length = m.end() - m.start()
            if length > starts.get(m.start(), -1):
                starts[m.start()] = length
    for name in sorted(known):
        if name in subjects:
            continue
        words = name_words[name]

        if any(m.end() - m.start() >= starts.get(m.start(), -1)
               for m in _occurrences(words)):
            recovered.append(name)
    subjects.extend(recovered)
    if recovered:
        report["recovered_subjects"] = [
            {"concept": n,
             "note": "the question names this concept; the decomposition "
                     "did not list it"} for n in recovered]

    def winner(hit: Mapping[str, Any], key: str) -> str | None:
        """The top-ranked candidate, or None -- `matches` is score-ordered.

        There is no top-level winner field on these payloads: `find_metric`
        reports only `matches`, and `find_attribute`'s `concept` is the INPUT
        filter rather than the owning concept. Reading either as the answer
        silently resolves nothing, which is how that went unnoticed until
        `test_skillfile` asserted a span had actually resolved.
        """
        rows = hit.get("matches") or ()
        return rows[0].get(key) if rows else None

    definitions: list[str] = []
    for phrase in spans("definitional_phrases"):
        hit = invoke("find_metric", _undetermined(phrase))
        name = winner(hit, "metric")
        if hit.get("status") in ("unique", "partial") and name:
            if name not in definitions:
                definitions.append(name)
        else:
            report["unresolved"].append({"slot": "definitional", "phrase": phrase})

    metrics: list[str] = []

    enumerated_operands: list[str] = []
    enumerated_attributes: list[str] = []
    metric_routes: dict[str, str] = {}
    for phrase in spans("quantity_phrases"):
        hit = invoke("find_metric", _deanaphored(phrase))
        status, name = hit.get("status"), winner(hit, "metric")
        if status == "ambiguous":
            report["ambiguous"].append(
                {"slot": "quantity", "phrase": phrase,
                 "between": [m.get("metric") for m in (hit.get("matches") or ())[:3]]})
        elif status in ("unique", "partial") and name:

            if _AVERAGE_LEAD.search(phrase):
                report.setdefault("averages", []).append(
                    {"phrase": phrase,
                     "note": "an average is a combine (divide) or over: avg; "
                             "the parts side by side do not compute it"})
            if (_ENUMERATION_LEAD.match(phrase)
                    and not _AGGREGATE_SIGNAL.search(question or "")):
                entry = skills.ontology.metrics.get(name)
                op = getattr(entry, "operand", None) if entry else None
                operand = getattr(op, "concept", None)
                attribute = getattr(op, "attribute", None)
                if operand and attribute:
                    if operand not in subjects:
                        subjects.append(operand)
                        enumerated_operands.append(operand)
                    qualified = f"{operand}.{attribute}"
                    if qualified not in enumerated_attributes:
                        enumerated_attributes.append(qualified)
                    for subject in list(subjects):
                        for route_entry in (data(invoke("find_metric", name,
                                                        subject=subject))
                                            .get("routes_from_subject") or ()):
                            for rid in route_entry.get("compatible_route_ids") or ():
                                metric_routes.setdefault(rid, name)
                    report.setdefault("enumerations", []).append({
                        "metric": name, "concept": operand,
                        "attribute": qualified,
                        "note": f"'{phrase}' enumerates {operand}: one row per "
                                f"{operand}, {qualified} as a dimension on one "
                                f"of that metric's governed routes. No measure "
                                f"-- a measure aggregates, and nobody asked for "
                                f"a total"})
                else:
                    metrics.append(name)
            else:
                metrics.append(name)

            entry_metric = skills.ontology.metrics.get(name)
            if entry_metric is not None and entry_metric.is_composite:
                bases = skills.ontology.measured_over(name)
                if len(bases) == 1:
                    for subject in subjects:
                        if subject == bases[0]:
                            continue
                        for path in (data(invoke("find_paths", subject, bases[0]))
                                     .get("paths") or ()):
                            metric_routes.setdefault(path["route_id"], name)
            else:
                for subject in subjects:
                    for route_entry in (data(invoke("find_metric", name, subject=subject))
                                        .get("routes_from_subject") or ()):
                        for rid in route_entry.get("compatible_route_ids") or ():
                            metric_routes.setdefault(rid, route_entry.get("metric") or name)
            if hit.get("unmatched_words"):
                report.setdefault("partial", []).append(
                    {"slot": "quantity", "phrase": phrase,
                     "unmatched": hit["unmatched_words"]})
        else:
            report["unresolved"].append({"slot": "quantity", "phrase": phrase})

    def _selectable(concept: str, attr: str) -> bool:
        """Does this attribute have a backing column?

        An attribute the mapping gives no column cannot appear in SQL, so it is
        not a candidate for anything -- yet it was scoring as one.
        `PolicyHolder.name` maps to None and was tying with `PolicyHolder.id`,
        which is a tie between a real column and a thing that does not exist.
        """
        try:
            spec = skills.ontology.concept(concept).attributes[attr]
        except Exception:                       # noqa: BLE001
            return False
        return getattr(spec, "column", None) is not None

    def _names_a_concept(phrase: str) -> str | None:
        """The concept this phrase IS, if it is one.

        `policy holder`, `catastrophe` and `agent` are CONCEPTS, and were being
        matched against attribute labels -- where each collides with every `id`
        and `name` in the graph and comes back ambiguous. A phrase that names a
        concept is not an ambiguous attribute; it is a concept, and the ontology
        declares which attribute identifies one.

        A PLURAL NAMES THE CONCEPT TOO. "Return policy holders and the claims
        they have made" names two concepts, and neither matched: the comparison
        was exact, so `claims` missed `Claim` and `policy holders` missed
        `PolicyHolder`. Both then fell through to attribute matching, where
        `claim` scores against the LABELS `claim open date` and `claim close
        date` and comes back ambiguous between two dates -- so a question asking
        for claims resolved to nothing that identifies one, and the plan's
        `Claim.id` was rejected as never resolved.

        Only a trailing `s`, or an `-ies` written back to `-y`, and only when
        the singular is itself a concept name. That cannot invent a match: if
        the shortened form is not declared, nothing is returned.

        `-ies` WAS MISSING AND IT MATTERED. "policies" shortens to "policie",
        which is not a concept, so the question fell through to a general
        attribute search -- where "polic..." matches Policy, PolicyHolder,
        PolicyAmount and PolicyCoverageDetail equally, and the phrase came back
        AMBIGUOUS. On query-e610253b that left the whole policy side of the
        question resolving to nothing, the plan named `Policy.id` anyway, and
        review refused it twice for naming what resolution never settled: no
        SQL on either run. One plural form, four concepts sharing a prefix.
        """

        words = [w for w in phrase.casefold().replace("_", " ").split() if w]
        while words and words[0] in _DETERMINERS:
            words.pop(0)
        want = "".join(words) or phrase.casefold().replace(" ", "")
        names = {concept.casefold().replace("_", ""): concept
                 for concept in skills.ontology.concept_names()}
        if want in names:
            return names[want]
        if want.endswith("ies") and want[:-3] + "y" in names:
            return names[want[:-3] + "y"]
        if want.endswith("s") and want[:-1] in names:
            return names[want[:-1]]
        return None

    def _candidate(qualified: str) -> dict:
        """What a tied candidate actually IS -- enough to settle it here.

        An ambiguity that reports only names forces whoever reads it to go and
        look, which costs a tool, a turn, and (measured) a wandering agent. The
        facts that decide these ties are three fields the ontology already holds,
        so they travel WITH the tie:

          column      what backs it. None means it cannot appear in SQL at all,
                      so it was never a real candidate.
          is_display  the ontology's own statement that this attribute is what
                      identifies its concept.
          concept     which concept it belongs to, so a tie ACROSS concepts is
                      visibly different from a tie within one.
        """
        try:
            concept, attr = qualified.split(".", 1)
            c = skills.ontology.concept(concept)
            spec = c.attributes[attr]
        except Exception:                       # noqa: BLE001
            return {"attribute": qualified, "column": None, "selectable": False}
        column = getattr(spec, "column", None)
        return {"attribute": qualified, "concept": concept, "column": column,
                "selectable": column is not None,
                "is_display": c.display == attr,
                "label": getattr(spec, "label", None)}

    concepts: list[str] = []
    concepts.extend(o for o in enumerated_operands if o not in concepts)
    attributes: list[str] = []
    attributes.extend(enumerated_attributes)
    for phrase in spans("attribute_phrases"):

        named = _names_a_concept(phrase)

        if (named and metrics and subjects and named == subjects[0]
                and (phrase.split() or [""])[0].casefold() in ("each", "every")
                and not getattr(skills.ontology.concept(named), "grain", None)):
            report.setdefault("grain_phrases", []).append(
                {"phrase": phrase, "concept": named,
                 "note": "names the subject itself in an aggregating question; "
                         "the subject already fixes what each row is about, so "
                         "this span scopes the question and is not a column to "
                         "project"})
            continue
        if named:
            display = getattr(skills.ontology.concept(named), "display", None)
            if display and _selectable(named, display):
                concepts.append(named)
                attributes.append(f"{named}.{display}")
                continue
        best = None
        tied: list[dict] = []
        unscoped = None
        for scope in [*subjects, None]:
            candidate = invoke("find_attribute", phrase, concept=scope)

            _owner = (candidate.get("matches") or [{}])[0].get("concept")
            _attr = str(candidate.get("attribute") or "").split(".")[-1]
            if _owner and _attr and not _selectable(_owner, _attr):
                candidate = {**candidate, "status": "absent"}
            if scope is None:

                unscoped = candidate
            if candidate.get("status") not in ("unique", "partial"):
                continue
            rank = (len(candidate.get("unmatched_words") or ()),
                    -((candidate.get("matches") or [{}])[0].get("score") or 0))
            if best is None or rank < best[0]:
                best = (rank, candidate)
                tied = [candidate]
            elif rank == best[0]:

                tied.append(candidate)

        hit = best[1] if best else (unscoped or {})

        distinct = {str(c.get("attribute")) for c in tied if c.get("attribute")}
        if len(distinct) > 1:
            names = sorted(distinct)

            owners = {n.rpartition(".")[0] for n in names}
            named = {str(s) for s in (decomp.get("subjects") or [])}
            if owners and owners <= named:
                for qualified_name in names:
                    concept_owner = qualified_name.rpartition(".")[0]
                    concepts.append(concept_owner)
                    attributes.append(
                        _prefer_display(skills, concept_owner, qualified_name))
                continue

            report["ambiguous"].append(
                {"slot": "attribute", "phrase": phrase, "between": names,
                 "candidates": [_candidate(n) for n in names],
                 "why": "several subjects matched this phrase equally well"})
            continue

        status = hit.get("status")
        owner, qualified = winner(hit, "concept"), hit.get("attribute")
        if status == "ambiguous":
            names = [m.get("attribute") for m in (hit.get("matches") or ())[:4]]

            owners = {str(n).rpartition(".")[0] for n in names if n}
            named = {str(s) for s in (decomp.get("subjects") or [])}
            if owners and owners <= named and len(owners) > 1:
                for qualified_name in names:
                    if not qualified_name:
                        continue
                    concept_owner = str(qualified_name).rpartition(".")[0]
                    concepts.append(concept_owner)
                    attributes.append(_prefer_display(
                        skills, concept_owner, str(qualified_name)))
                continue

            report["ambiguous"].append(
                {"slot": "attribute", "phrase": phrase, "between": names,

                 "candidates": [_candidate(n) for n in names if n]})
        elif status in ("unique", "partial") and owner:
            concepts.append(owner)
            if qualified:
                attributes.append(qualified)
        else:
            report["unresolved"].append({"slot": "attribute", "phrase": phrase})

    if metrics:
        for subject in subjects[:1]:
            try:
                spec = skills.ontology.concept(subject)
            except Exception:                   # noqa: BLE001
                continue
            display = getattr(spec, "display", None)
            if (not getattr(spec, "grain", None) or not display
                    or not _selectable(subject, display)
                    or any(q.rpartition(".")[0] == subject for q in attributes)):
                continue
            concepts.append(subject)
            attributes.append(f"{subject}.{display}")
            report.setdefault("grain_identity", []).append({
                "concept": subject, "attribute": f"{subject}.{display}",
                "note": f"{subject} is {spec.grain}; an aggregate anchored on "
                        f"or reaching it must project this attribute so each "
                        f"row says which {subject} it is about"})
    elif concepts:

        for concept in list(dict.fromkeys([*concepts, *subjects])):
            try:
                spec = skills.ontology.concept(concept)
            except Exception:                   # noqa: BLE001
                continue
            display = getattr(spec, "display", None)
            if (not display or not _selectable(concept, display)
                    or any(q.rpartition(".")[0] == concept for q in attributes)):
                continue

            if any(q.rpartition(".")[0] == concept
                   for q in enumerated_attributes):
                continue
            attributes.append(f"{concept}.{display}")
            report.setdefault("listing_identity", []).append({
                "concept": concept, "attribute": f"{concept}.{display}",
                "note": f"a listing names {concept}; each row must say which "
                        f"{concept} it is, so its display is projectable"})

    literals: list[dict] = []
    for phrase in spans("literal_phrases"):
        for concept in subjects + concepts:
            for attr in _searchable(skills, concept):
                found = invoke("search_values", concept, attr, phrase)
                if found.get("values"):
                    literals.append({"phrase": phrase, "concept": concept,
                                     "attribute": attr,
                                     "values": list(found["values"])[:5]})
                    break
            else:
                continue
            break
        else:
            report["unresolved"].append({"slot": "literal", "phrase": phrase})

    import re as _re_t
    periods: list[dict] = []
    for phrase in spans("time_phrases"):
        bounds = _period(phrase)
        if not bounds:
            report["unresolved"].append({"slot": "time", "phrase": phrase})
            continue
        residue = " ".join(
            w for w in _re_t.split(r"[^\w-]+", phrase.casefold())
            if w and w not in _TIME_STOPWORDS
            and not _re_t.fullmatch(r"[\d-]+", w))
        hit = None
        for scope in list(dict.fromkeys(subjects + concepts)):
            dated = _date_attributes(skills, scope)
            if not dated:
                continue
            found = invoke("find_attribute", residue, concept=scope) if residue else {}
            name = str(found.get("attribute") or "")
            bare = name.split(".")[-1]
            if bare and bare in dated:
                hit = (scope, bare)
                break
            if not residue and len(dated) == 1:
                hit = (scope, dated[0])
                break
        if hit is None:
            report["unresolved"].append({"slot": "time", "phrase": phrase})
            continue
        concept_name, attribute = hit
        concepts.append(concept_name)
        attributes.append(f"{concept_name}.{attribute}")
        periods.append({"phrase": phrase, "concept": concept_name,
                        "attribute": attribute, "operator": "BETWEEN",
                        "value": [bounds[0], bounds[1]]})

    route_ids: list[str] = []
    routes: list[dict] = []
    anchors = list(dict.fromkeys(subjects + concepts))
    for src in anchors:
        for dst in anchors:
            if src == dst:
                continue
            for path in invoke("find_paths", src, dst).get("paths", ()):
                route_ids.append(path["route_id"])

                routes.append({"route": path["route_id"],
                               "via": path.get("path_id"),
                               "hops": path.get("hops"),
                               "fan_out": path.get("max_fan_out")})

    if len(routes) > ROUTE_DETAIL_CAP:
        report["routes_undescribed"] = len(routes) - ROUTE_DETAIL_CAP
        routes = routes[:ROUTE_DETAIL_CAP]
    report["routes"] = routes

    carried = [(rid, metric) for rid, metric in metric_routes.items()
               if rid != "SELF" and rid not in set(route_ids)]
    for rid, _metric in carried:
        route_ids.append(rid)
    if metric_routes:

        by_metric: dict[str, list[str]] = {}
        for rid, metric in carried:
            by_metric.setdefault(metric, []).append(rid)

        for entry in report.get("enumerations") or []:
            name = entry.get("metric")
            if name and name not in by_metric:
                mine = [rid for rid, metric in metric_routes.items()
                        if metric == name]
                if mine:
                    by_metric[name] = mine

        for name in dict.fromkeys(metrics):
            if name in by_metric:
                continue
            entry = skills.ontology.metrics.get(name)
            if entry is None or not entry.is_composite:
                continue
            bases = skills.ontology.measured_over(name)
            if len(bases) != 1:
                continue
            if bases[0] in subjects:
                metric_routes.setdefault("SELF", name)
            for rid in list(route_ids):
                if rid.split("#")[0].split(">")[-1] == bases[0]:
                    metric_routes.setdefault(rid, name)
            mine = [rid for rid, metric in metric_routes.items()
                    if metric == name]
            if mine:
                by_metric[name] = mine
        report["metric_routes"] = {
            "note": "each measure below is computed on a concept that no route "
                    "between your subjects reaches; give that measure one of "
                    "these routes, not a subject route",
            "by_metric": by_metric}

    composition = {m: list(_summands(skills, m)) for m in dict.fromkeys(metrics)}
    report["summands"] = {m: parts for m, parts in composition.items() if parts}

    import re as _re_d

    tail = _re_d.split(r"\b(?:which is|which are|where .{0,40}? is|defined as)\b",
                       question or "", maxsplit=1, flags=_re_d.I)
    if len(tail) > 1 and metrics:

        after = _re_d.split(r"\bby\b", tail[-1], maxsplit=1)[0].casefold()
        stated = [m for m in dict.fromkeys(metrics)
                  if not composition.get(m)
                  and _re_d.sub(r"[^a-z]", "", m.casefold())
                  in _re_d.sub(r"[^a-z]", "", after)]
        for m in stated:
            if m not in definitions:
                definitions.append(m)

    definitions = [d for d in definitions
                   if not (set(_summands(skills, d))
                           and set(_summands(skills, d)) <= set(definitions) - {d})]

    if definitions:
        report["definitions"] = {
            "metrics": definitions,
            "note": "named only to define another term; do not project these "
                    "unless the question asks to see them broken out"}

    report.update(subjects=subjects, metrics=metrics, concepts=concepts,
                  attributes=attributes, literals=literals, periods=periods,
                  off_question=off_question, route_count=len(route_ids),
                  resolver_calls=dict(calls))
    return {"subjects": subjects, "metrics": metrics, "concepts": concepts,
            "literals": literals, "periods": periods,
            "route_ids": route_ids, "report": report,
            "summands": report["summands"],
            "abstain_reason": (None if subjects
                               else _why_no_subject(decomp, unknown))}

def _main(argv: list[str]) -> int:
    import json as _json
    import sys as _sys
    if len(argv) < 2:
        print('usage: resolve.py \'<decomposition json>\' \'<question>\'', file=_sys.stderr)
        return 2
    _sys.path.insert(0, __file__.split("/workflow/")[0])
    from spc.skills import default_skills
    out = resolve(_json.loads(argv[0]), argv[1], default_skills())

    print(_json.dumps({"report": out["report"], "route_ids": out["route_ids"],
                       "subjects": out["subjects"], "metrics": out["metrics"],
                       "concepts": out["concepts"],
                       "abstain_reason": out["abstain_reason"]}, indent=2))
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
