# -*- coding: utf-8 -*-
"""
Every string that appears on either deck, transcribed verbatim from the
CMU BME Forum 2026 build brief. Import from here rather than retyping:
the numbers are quoted to the precision the brief specifies and the
wording is the author's, not the building agent's.

DECK_A  -> 3 slides, dense, read silently by a judge, exported to PDF
DECK_B  -> 9 slides, sparse, narrated, exported to video

Nothing in this file should be paraphrased, shortened or "improved".
"""

# ----------------------------------------------------------------- authors
AUTH_LINE = 'Ahaan Nigam¹, Rabira Tusi², Pulkit Grover², [Fourth Author]²'
AFF1 = '¹Independent researcher · ahaannigam27@gmail.com'
AFF2 = '²Carnegie Mellon University · rtusi@cmu.edu · pulkitgrover@cmu.edu'
# Presenting author is Ahaan Nigam and must be set in bold wherever the
# author line appears. [Fourth Author] is an unresolved placeholder: do not
# invent a name, an affiliation or an email. See HANDOFF.md §11.

FOOTER_SHORT = 'CMU BME Forum 2026 · Nigam, Tusi, Grover'
REPO_URL = '[repo URL]'   # unresolved placeholder, keep the brackets

# ================================================================= DECK A
DECK_A = [
    dict(
        n='A1',
        title=('Scheduling, not just dose: the order of sun exposure changes '
               'how much ultraviolet darker skin needs'),
        left_heading_1='Background',
        background=[
            "Dose-response models predict one 25(OH)D rise for one cumulative dose. "
            "Webb 2021's consolidated fit and Grigalavicius 2015's hyperbola both hold "
            "Fitzpatrick type, or a fixed melanin index, constant across the whole "
            "exposure course.",
            "A darker-skinned participant on a nine-week course is not the same optical "
            "system in week 9 as in week 1. The skin has tanned and the stratum corneum "
            "has thickened.",
            "Guidance therefore prescribes evenly spaced fixed-duration sessions. Nobody "
            "has tested whether the spacing itself is a free variable.",
        ],
        left_heading_2='Purpose',
        purpose=("Treat photoadaptation as a state that co-evolves with the schedule, and "
                 "find the optimal ordering of a fixed annual UV budget at a fixed "
                 "sufficiency floor, for Fitzpatrick types I–VI."),
        right='F1 — the three-state schematic, built natively as PowerPoint shapes',
        footer='Full model, parameter provenance and validation tiering: [repo URL]',
    ),
    dict(
        n='A2',
        title='A three-state model, an exactly-solved dose, and a one-parameter search over ordering',
        model_heading='Model',
        model_code=(
            "S = A · ln(1 + sed · E_source) · (bsa / BSA_REF)^0.5\n"
            "      · exp(−K_const · M_base(F))        constitutive pigment\n"
            "      · exp(−K_ATTEN · (M − M_base(F)))  facultative tan\n"
            "      · exp(−K_SC · SC)                  stratum corneum\n"
            "      · g(D),   g(D) = 1 / (1 + (D/56.0)²)   biphasic feedback"
        ),
        model_note=("Implemented as a hybrid pulse process, not a continuous integration: all "
                    "three states relax analytically between sessions at dt = 0.5 d, and each "
                    "session applies a saturating jump."),
        search_heading='Search',
        search=[
            "Session days warped by one monotone exponent, tᵢ = T·(i/N)^γ. γ = 1 recovers "
            "uniform spacing exactly, so the search cannot lose to uniform by being unable "
            "to express it.",
            "Per-session dose solved exactly by bisection on a provably monotone function, "
            "so every reported schedule sits precisely on the sufficiency floor and all "
            "comparisons are at matched efficacy.",
            "Differential evolution over the remaining parameters, across two dose families "
            "(two-phase, exponential).",
            "Constraints, none relaxed: per-session dose ≤ 70% MED; the 50 nmol/L floor "
            "holds every day after a 90-day ramp-in; no session longer than 120 min.",
        ],
        scenario_heading='Reference scenario',
        scenario=("UV index 6, geometry factor 0.35, 35% exposed body surface area, held "
                  "constant on every exposure day. A frozen reference condition, not a "
                  "calendar recommendation."),
        right_fig='f5_evidence_tiers.png',
        # The right-hand text sits UNDER F5. Bold spans are marked with ** ** and
        # must be rendered as real bold runs, not as literal asterisks.
        right_text=[
            "14 of 14 validation checks pass, and that count alone means little. Tiered by "
            "what each check actually tests, **two** are external quantitative comparisons "
            "and both carry circularity caveats.",
            "The one fully held-out test, Webb/Kift 2018, **fails**: modelled FST V : FST II "
            "dose ratio 3.01× against a reported 2.5–3.0×, missing the ceiling by 0.49%. The "
            "two matched-dose lamp studies bracketing that value disagree with each other by "
            "roughly 2×.",
        ],
        footer='Full model, parameter provenance and validation tiering: [repo URL]',
    ),
    dict(
        n='A3',
        title=('Ordering a fixed UV budget is worth 30–36% of the dose, and the saving grows '
               'with pigmentation'),
        panels=[
            dict(fig='f3_frontload_saving.png',
                 lead='Front-loading saving, 365 sessions.',
                 caption=("Same session count, same erythema ceiling, same sufficiency floor. "
                          "Only the order changes. −29.56% at FST I rising monotonically to "
                          "−35.50% at FST VI.")),
            dict(fig='f2_feasibility_heatmap.png',
                 lead='What is reachable at all.',
                 caption=("Cumulative UV needed to reach and hold 50 nmol/L. Grey cells have no "
                          "feasible schedule. 9 of 36 are infeasible under uniform spacing; "
                          "reordering alone rescues 2.")),
            dict(fig='f4_two_cancer_axes.png',
                 lead='The two cancer axes diverge.',
                 caption=("Front-loading raises cumulative dose 4.43% but cuts peak per-session "
                          "dose 28.16%. SCC risk scales as dose^1.7, intermittent-burn risk as "
                          "dose³, so the intermittency index falls a median 21.36% while SCC "
                          "rises a median 7.65%.")),
        ],
        conclusion=[
            "Ordering a fixed annual UV budget is worth 30–36% of the total dose at a fixed "
            "sufficiency floor, and the saving is largest for the darkest skin.",
            "The same reordering lowers the intermittency-driven cancer index, because it cuts "
            "peak per-session dose faster than it raises cumulative dose.",
            "Absolute minutes are not yet a dosing instruction. The transferable output is the "
            "ranking.",
        ],
        footer=("Uncalibrated priors on SC_max, alpha_SC and K_SC mean absolute 25(OH)D levels "
                "carry wider uncertainty than the rankings shown. Code and parameter "
                "provenance: [repo URL]"),
    ),
]

# ================================================================= DECK B
# 'lines' are the words that go ON the slide. 'notes' is the narration,
# verbatim from §5 of the brief, and belongs in the speaker notes.
# 'subs' are OPTIONAL one-line glosses used by the style-3 layout; they are
# not in the brief. Drop them if you want the sparser reading.

DECK_B = [
    dict(
        n='V1', t='0:00–0:20', fig=None,
        title='Scheduling, not just dose',
        sub='The order of sun exposure changes how much UV darker skin needs',
        lines=[], subs=[],
        notes=("Public health guidance tells you to get regular sun for vitamin D. It tells you "
               "roughly how long. It never tells you when. I'm going to show you that for the "
               "same annual amount of ultraviolet, the order of the sessions is worth about a "
               "third of the total dose, and that the saving is largest for the darkest skin."),
    ),
    dict(
        n='V2', t='0:20–0:50', fig=None,
        title='Guidance says how long. It never says when.',
        lines=['Deficiency tracks pigmentation',
               'Standard answer: longer sessions, evenly spaced',
               'Evenly spaced is an assumption, not a result'],
        subs=[], full_width=True,
        notes=("Vitamin D deficiency tracks skin pigmentation. Darker skin needs more ultraviolet "
               "for the same serum 25-hydroxyvitamin D, and standard guidance handles that by "
               "prescribing longer sessions, evenly spaced. Evenly spaced is an assumption nobody "
               "has tested. It's the default because it's simple, not because it's optimal."),
    ),
    dict(
        n='V3', t='0:50–1:20', fig='f7_fixed_vs_coupled.png',
        title="Existing models can't answer it",
        lines=['One dose → one rise',
               'Skin type held constant',
               'Adaptation delays sufficiency from day 84 to day 147'],
        subs=['Webb 2021, Grigalavicius 2015',
              'the same optical system in week 9 as in week 1',
              'identical FST VI schedule, identical UV budget'],
        notes=("The reason it hasn't been tested is that the models we have are single-exposure "
               "fits. Webb's consolidated dose-response and Grigalavicius's hyperbola both give "
               "you one rise in serum D for one cumulative dose, and both hold skin type, or a "
               "fixed melanin index, constant across the whole course. But a darker-skinned "
               "person on a nine-week schedule is not the same optical system in week nine as in "
               "week one. Their skin has tanned and their stratum corneum has thickened. Here are "
               "two runs of the same type-six schedule, same dose, same days. Freeze the skin and "
               "you reach sufficiency on day eighty-four. Let it adapt and you don't get there "
               "until day one forty-seven. If pigmentation is a state that the schedule itself "
               "changes, then the schedule has to be a decision variable, not a fixed input."),
    ),
    dict(
        n='V4', t='1:20–1:55', fig='F1_NATIVE',
        title='Three coupled states',
        lines=['Serum 25(OH)D', 'Melanin index', 'Stratum corneum'],
        subs=['what the schedule has to hold above 50 nmol/L',
              'tans under UV, then suppresses the next session',
              'thickens under UV, on a slower clock'],
        notes=("So we built a three-state model: serum 25-hydroxyvitamin D, a melanin index, and "
               "stratum corneum thickness, all coupled. Ultraviolet drives synthesis, and the "
               "same ultraviolet drives the two attenuating states that suppress future "
               "synthesis. Synthesis is logarithmic in dose, scaled by exposed body surface area, "
               "attenuated multiplicatively by constitutive pigment, by the facultative tan, and "
               "by the stratum corneum, then inhibited by homeostatic feedback from circulating D "
               "itself. Between sessions all three states relax analytically; each session "
               "applies a saturating jump."),
    ),
    dict(
        n='V5', t='1:55–2:25', fig='f6_gamma_warp.png',
        title='One exponent controls the ordering',
        lines=['γ = 1 is exactly uniform',
               'Dose solved by bisection → matched efficacy',
               '3 constraints, none relaxed'],
        subs=['tᵢ = T·(i/N)^γ, so the search cannot lose to uniform',
              'every schedule sits precisely on the 50 nmol/L floor',
              '≤ 70% MED, floor holds after day 90, ≤ 120 min per session'],
        notes=("To ask about ordering, we warp the session days by a single exponent. Gamma of one "
               "is exactly uniform spacing, so the search can never lose to uniform by being "
               "unable to express it. Gamma above one bunches sessions early. For every schedule "
               "we solve the per-session dose exactly by bisection, so every candidate sits "
               "precisely on the fifty nanomole sufficiency floor and every comparison is at "
               "matched efficacy. Three constraints, none relaxed: no session above seventy "
               "percent of the individual minimal erythemal dose, the floor holds every day after "
               "a ninety-day ramp-in, and no session longer than two hours."),
    ),
    dict(
        n='V6', t='2:25–2:55', fig='f3_frontload_saving.png',
        title='Same sessions. Same limits. 30–36% less UV.',
        lines=['−29.56% at FST I', '−35.50% at FST VI', '−42.41% at a 183-session budget'],
        subs=['lightest skin', 'darkest skin', 'tighter budget, larger saving'],
        notes=("Here's the result. Same number of sessions, same erythema ceiling, same "
               "sufficiency floor. Only the order changes. Front-loading cuts the cumulative "
               "ultraviolet required by twenty-nine and a half percent at Fitzpatrick type one, "
               "and thirty-five and a half percent at type six. The saving grows monotonically "
               "with pigmentation, and at tighter session budgets it reaches forty-two percent "
               "for type six."),
    ),
    dict(
        n='V7', t='2:55–3:20', fig='f2_feasibility_heatmap.png',
        title='Ordering changes what is possible at all',
        lines=['9 of 36 infeasible under uniform spacing',
               'Reordering rescues 2',
               'The floor binds at day 90, not at year end'],
        subs=['FST × session budget', 'free spacing leaves 7 of 36 infeasible',
              'uniform spacing is still climbing at the ramp-in boundary'],
        notes=("Ordering also changes what's possible at all. Under uniform spacing, nine of "
               "thirty-six combinations of skin type and session budget have no schedule that "
               "reaches and holds sufficiency without burning and without a session nobody would "
               "sit through. Reordering alone rescues two of them. The constraint binds at the "
               "ramp-in boundary, not at the end of the year: uniform spacing is still climbing "
               "at day ninety and misses the floor by a fraction of a nanomole."),
    ),
    dict(
        n='V8', t='3:20–3:45', fig='f4_two_cancer_axes.png',
        title='Not a safety trade',
        lines=['Cumulative dose +4.43%', 'Peak per-session dose −28.16%',
               'Intermittency index −21.36%'],
        subs=['SCC risk scales as dose^1.7', 'intermittent-burn risk scales as dose³',
              'median across 22 paired feasible operating points'],
        notes=("And it isn't a safety trade. Front-loading raises total annual dose about four "
               "percent, but it cuts peak per-session dose by twenty-eight percent. Since "
               "squamous cell risk scales as a one-point-seven power of cumulative dose while "
               "intermittent-burn risk scales cubically with per-session dose, the "
               "intermittency-driven index falls by a median twenty-one percent. The two axes "
               "move in opposite directions, which is exactly why we report them separately "
               "instead of hiding it inside one combined number."),
    ),
    dict(
        n='V9', t='3:45–4:30', fig='f5_evidence_tiers.png',
        title='What this does and does not show',
        lines=['14/14 checks — 2 are external and quantitative',
               'The one held-out test fails',
               "The ranking transfers; the minutes don't yet"],
        subs=['four internal, three direction-only, three circular, two not vitamin-D evidence',
              'Webb/Kift 2018, FST V : FST II, 3.01× against a reported 2.5–3.0×',
              'stratum-corneum priors are uncalibrated'],
        notes=("Now the limits, because they matter more than the result. Our pipeline reports "
               "fourteen of fourteen validation checks passing, and that number is close to "
               "meaningless on its own. We tier every check by what it actually tests: four are "
               "internal self-consistency, three are direction-only, three are circular against "
               "data we fitted to, and two are labelled in their own source as not vitamin D "
               "evidence. Two are genuine external quantitative comparisons. Our one fully "
               "held-out test fails, by half a percent, sitting between two published lamp "
               "studies that disagree with each other by a factor of two. Absolute minutes are "
               "not a dosing instruction yet. What transfers is the ranking: the order of "
               "exposure is a free variable, it's worth about a third of the dose, and it helps "
               "the people who need the most sun the most. Thank you."),
    ),
]
