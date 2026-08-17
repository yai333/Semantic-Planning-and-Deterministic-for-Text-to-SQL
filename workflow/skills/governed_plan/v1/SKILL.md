---
name: governed-plan
description: Answer an analytics question over the governed ACME insurance ontology by resolving its words and building a plan. Use when a question needs data from the warehouse — claims, policies, premiums, losses, policy holders, agents.
metadata:
  tier: main

  # THE DETERMINISTIC STEPS THE AGENT MAY RUN, and the only things it can.
  #
  # It chooses WHICH step and supplies typed arguments; the engine builds the
  # invocation and runs it with the project interpreter. There is no command for a
  # model to compose and no step it can name that is not here.
  steps:
    subjects:
      script: scripts/subjects.py
      entry: subjects
      for: List the governed subjects you may anchor on, each with one line describing it, and the governed measure names. Call this first.
      args:
        question: {type: string, for: the question, verbatim}

    resolve:
      script: scripts/resolve.py
      entry: resolve
      for: Resolve verbatim spans of the question against the ontology. Returns what resolved, what did not, and the governed routes between your subjects.
      args:
        decomposition: {type: object, for: the question cut into slots, schema: schema}
        question: {type: string, for: the question, verbatim}



    # WHAT A ROUTE IS, when the id is not enough to choose by. `resolve` returns
    # ids -- "Claim>PolicyAmount#2" -- and two routes between the same pair can
    # mean different things: one reaches the coverage a claim was made against,
    # another detours through the insured object and collects every coverage on
    # it. The ordinal says neither. Read-only: it reports the ontology, adds no
    # vocabulary, and a plan still cannot name anything resolve did not return.
    describe_routes:
      script: scripts/describe_routes.py
      entry: describe_routes
      for: "Show every governed route between two concepts with its EDGE CHAIN and where it fans out. Call this when `resolve` offered several routes to the same concept and the ids alone do not say which relationship you want."
      args:
        source: {type: string, for: the concept the route leaves}
        target: {type: string, for: the concept it must reach}

    review_plan:
      script: scripts/review_plan.py
      entry: review
      for: "Check a draft plan against what the question asked and what resolution found. Call this on EVERY question, after drafting the plan and before submitting it -- it is the only review there is. Deterministic -- no model, no cost."
      args:
        plan: {type: object, for: the draft plan}
        resolved: {type: object, for: the whole output of resolve}

  # THE AGENT'S ANSWER, as structured output rather than another step.
  #
  # Its schema is built per question by `plan_schema` from what `resolve`
  # returned, so every route, metric and attribute the agent may name is one the
  # resolver produced -- a plan naming something unretrieved is not rejected
  # afterwards, it is unrepresentable.
  #
  # WHAT HAPPENS NEXT IS NOT THE AGENT'S BUSINESS. The engine compiles the plan
  # and the checker certifies its grain. Those were steps in this skill and are
  # not: they are not things the agent does, and a procedure describing them
  # told it to act where it has no move.
  #
  # THERE IS NO SECOND AGENT. A critic model used to read the finished plan and
  # pass or fail it. It failed plans that were correct -- it saw the question
  # and the plan but not what resolution had decided, so it re-argued settled
  # points from less information. Review is `review_plan` below: deterministic,
  # inside this skill, and able to compare the plan against the resolution.
  answer:
    name: submit_plan
    for: Submit the finished plan. Name only what resolve returned.
    schema: script:plan_schema

    # THE ORDER THIS WORKFLOW REQUIRES, declared here because the workflow is
    # this file's business and not the engine's. `requires` names steps that
    # must have run SINCE THE LAST SUBMISSION for the answer form to exist; the
    # engine reads the list and knows nothing about which steps they are.
    #
    # Without it the planner reviewed once, submitted, was refused, and
    # submitted again without reviewing -- two paid turns spent discovering what
    # the free deterministic step had already been told to say. The body says to
    # review before submitting; saying it was not enough, so it is declared.
    # `resolve` is NOT listed: the answer form is built FROM its output, so
    # resolution is already a precondition of the form existing at all. Listing
    # it here was self-defeating -- the set is emptied at each submission, and
    # resolve does not re-run after a review refusal, so the gate could never be
    # satisfied a second time.
    requires: [review_plan]

    # `on_compile_failure: resolve` USED TO BE DECLARED HERE AND WAS NEVER READ.
    # Nothing in the engine looked at it; the key appeared in this file and in a
    # review_plan docstring that described the behaviour as fact. Dead
    # configuration is worse than none, because the next reader reasons from it.
    #
    # What actually happens is stronger and needs no key: ANY refusal ends the
    # attempt, and the retry is a fresh conversation from the question, so it
    # re-decomposes and re-resolves whatever the failure was. The behaviour the
    # key promised is the behaviour every refusal already gets.

    # And which step the ENGINE re-runs on whatever is actually submitted. The
    # agent calls `review_plan` on a draft of its choosing; this is the same
    # step run on the plan that will actually compile, and an `error` finding
    # refuses it. Declared, so the engine names no step of its own.
    gate: review_plan

  schema: decompose.schema.json
---
# Governed plan

You answer by resolving the question's own words and then submitting a **plan**.
You never write SQL: the plan is compiled for you, and a plan naming anything
that was not retrieved cannot even be expressed.

Each step is a tool. You choose which to call and with what; there are no
commands to write and nothing runnable that is not listed here.

## 1. `subjects` — see what you may anchor on

Call `subjects` with the question. You get every governed subject with one line
describing it, and the names of the governed measures. **That list is the whole
of what you may anchor on.** If nothing on it fits, the question is out of scope:
say so and stop.

## 2. `resolve` — resolve the question's own words

Cut the question into **verbatim spans** — its own words, not ontology
vocabulary — and call `resolve` with them and the question:

```json
{"subjects": ["Claim", "Policy"],
 "quantity_phrases": ["how many claims"],
 "definitional_phrases": [],
 "attribute_phrases": ["policy number"],
 "literal_phrases": [], "comparison_phrases": [],
 "rank_phrases": [], "time_phrases": []}
```

### A phrase the question only uses to DEFINE something

Most slots are about what KIND of thing a span names. `definitional_phrases` is
about the span's ROLE in the sentence, and only the sentence settles it — the
same words are a quantity in one question and a definition in another.

A question may name several things in order to say what ONE thing means:

- *"X by C, where X is A plus B"* — `X` is the quantity; `A` and `B` are
  definitional. The clause says what `X` MEANS.
- *"A and B by C"* — both are quantities. Nothing is definitional.

**The test: would the question still be fully answered without showing it?** If
yes, it is definitional.

A definitional span is still resolved — the term it defines is computed from it —
but it is not something your plan must produce, and the answer schema will not
let you project it.

## 2b. `describe_routes` — when the route ids do not say which one you want

`resolve` returns routes as ids — `A>B#1`, `A>B#2` — and the ordinal says
nothing about what the route MEANS. Two routes between the same pair of concepts
can follow entirely different relationships, and picking the wrong one answers a
different question while compiling perfectly.

Call `describe_routes` with the two concepts when you have a choice to make. You
get each route's EDGE CHAIN and where it fans out, shortest first.

Read it this way:

- **Same final edge, different lengths** — they reach the same concept by the
  same relationship, so the extra legs can only widen what is counted. Take the
  shortest.
- **Different final edges** — different relationships, and the choice is
  semantic. Read the edge names and pick the one the question describes.
- **`fans_out`** — one flag per step. Each `true` is a step that may reach
  several rows, so more of them means more multiplication.

Use a `route` value from that list **verbatim**. The ids are not composable and
you cannot construct one: a route you invent will be refused.

## 3. `review_plan` — check your draft before you submit it

Draft the plan, then call `review_plan` with it and the resolution. **Do this on
every question**, not only the doubtful ones: it is deterministic, it costs
nothing and it makes no model call, and it is the only review there is.

Pay particular attention when `resolve` reported anything under **`ambiguous`**
or **`unresolved`** — part of the question did not make it into your vocabulary,
and a plan built on the rest silently answers something smaller.

It is deterministic and costs nothing, and it answers two things nothing else
checks in time to be useful:

1. **Does your plan use what was retrieved for it?**
2. **Can the rows your plan asks for exist together?**

It returns `{"ok": ..., "findings": [...]}`. An `error` means the plan ignores
something resolution found for it — a metric the question named, an attribute it
asked to see, a route that was never retrieved — or that it asks for a shape the
compiler cannot certify.

### Two branches that each reach several rows cannot share a row

Some routes reach at most one row from the subject; others may reach several.
Project **two dimensions on routes that each reach several, and that leave the
subject by genuinely different paths**, and every output row becomes one
arbitrary pairing of the two — a row that answers nothing. The finding names
both, and the compiler would otherwise refuse the plan afterwards in terms of
SQL aliases you never wrote.

A route that continues another is not a second branch: if one path is the first
part of the other, both projections sit on one chain and there is nothing wrong.

**Do not answer it by deleting a dimension the question asked for.** That
compiles, and answers a smaller question. Ask for one of them as a MEASURE
instead, so it is aggregated at its own grain — or, if the question genuinely
wants both as columns and neither can be aggregated, say that it cannot be
answered at one grain and why.

**Fix what it names and review again.** A plan that omits a resolved attribute is
legal, certifiable, and answers a smaller question than the one asked — the
compiler and the checker cannot tell, so this is the only place it is caught.

Where a finding is a `warn` — an ambiguous phrase, a span that resolved to
nothing — you are not required to act, but you are required not to pretend it did
not happen: say so in your answer rather than dropping the phrase.

## 4. `submit_plan` — your answer

Submit the plan. Name only what `resolve` returned:

```json
{"subject": "Claim",
 "measures":   [{"metric": "ClaimCount", "route": "SELF"}],
 "dimensions": [{"attribute": "policy_number", "route": "Claim>Policy#1"}]}
```

- `subject` — the grain: what one output row is about
- `measures` — a governed metric by name, and the route that reaches it. **This
  may be empty.** A question that asks to LIST or RETURN rows — "return A and B
  and the C for each" — has no quantity in it, and a plan with no measures
  returns exactly those rows. Inventing a metric to fill the slot changes the
  answer from the rows that were asked for into a count of them, and every
  layer after you will accept it
- `over` — set beside a metric when the question asks for an AVERAGE, a MINIMUM
  or a MAXIMUM of a metric that is itself built from others. A metric's own
  definition already says how to total it, so a total needs nothing here; the
  other aggregates do, because they cannot be assembled from the parts'
  aggregates. The field only appears when some metric can carry it
- `dimensions` — what to group by, and the route that reaches it
- `filters` — what RESTRICTS the rows, when the question restricts them. It
  appears only when your decomposition quoted a phrase that narrows the
  question — a literal, a comparison, or a time span — and its absence means you
  did not say the question filters. A clause naming WHEN something happened
  restricts the rows exactly as one naming HOW MUCH does, so quote it in
  `time_phrases` and resolution returns it under `periods`, already bounded and
  already attached to the attribute it bounds: copy that straight in. Dropping
  the restriction answers a larger question than the one asked, and on a small
  fixture it may return the same rows and look right. **A column you filter on
  is not thereby a column to show** — restrict by it, and project only what the
  question asks to see
- `route` — `SELF` for the subject's own columns, otherwise a `route` from step 2

If several readings are equally supported, say so rather than choosing one.

Submit after `review_plan` returns `ok` — or, for a `warn`, after deciding it
genuinely does not apply and saying which.

**An `error` finding is not advice.** The engine re-runs `review_plan` on
whatever you actually submit, and an `error` ends the attempt. There is no
"say why and submit anyway" for one: explaining an error finding does not clear
it, and a plan that carries one cannot be certified however well argued. If an
error will not clear, the fix is a DIFFERENT PLAN — change the subject you
anchor on, the route, or the projection. Every error finding carries a
`recommend` naming a move that is actually available.

**That is the end of your work.** The plan is compiled and its grain certified
after you. If it comes back refused, that attempt is over: you do not get to
edit this plan. The run starts again from the question, carrying only the
reason it failed — so a retry re-decomposes and re-resolves from scratch, and
cutting the question differently is the point of it, not a waste.
