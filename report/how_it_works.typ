// ============================================================================
// How the FPL System Actually Works - illustrated plain-language walkthrough
// ============================================================================

// ---- design system ----------------------------------------------------------
#let navy = rgb("#16324f")
#let teal = rgb("#1e7d63")
#let amber = rgb("#c98a1c")
#let brick = rgb("#b04a35")
#let ink = rgb("#222222")
#let paper-gray = rgb("#f4f5f7")
#let mid-gray = rgb("#8a929c")

#set page(margin: (x: 2.3cm, y: 2.5cm), numbering: "1", footer: context [
  #set text(size: 8.5pt, fill: mid-gray)
  #line(length: 100%, stroke: 0.4pt + mid-gray.lighten(40%))
  #v(-0.4em)
  How the FPL System Actually Works #h(1fr) #counter(page).display("1")
])
#set text(font: "New Computer Modern", size: 10.7pt, fill: ink, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [
  #v(1.1em)
  #text(fill: navy, size: 14pt, weight: "bold")[#it]
  #v(0.1em)
  #line(length: 100%, stroke: 1.2pt + navy.lighten(60%))
  #v(0.3em)
]
#show heading.where(level: 2): it => [
  #v(0.7em)
  #text(fill: navy.lighten(15%), size: 11.5pt, weight: "bold")[#it]
  #v(0.15em)
]
#show figure.caption: it => [
  #set text(size: 9pt, fill: mid-gray, style: "italic")
  #it
]

// Callout box for key ideas.
#let keyidea(title, body) = block(
  width: 100%, inset: 11pt, radius: 4pt,
  fill: teal.lighten(88%), stroke: (left: 2.5pt + teal),
)[
  #text(fill: teal.darken(20%), weight: "bold", size: 10pt)[#title]
  #v(0.25em)
  #text(size: 10.2pt)[#body]
]

// Horizontal bar chart helper.
#let hbars(rows, max-val, track: 8.6cm, label-w: 4.6cm, bar-h: 13pt) = {
  set text(size: 9.3pt, hyphenate: false)
  set par(justify: false)
  stack(
    spacing: 5pt,
    ..rows.map(((label, val, col, note)) => grid(
      columns: (label-w, track, auto),
      column-gutter: 8pt,
      align: (right + horizon, left + horizon, left + horizon),
      text(size: 9.3pt)[#label],
      box(width: track, height: bar-h, fill: paper-gray, radius: 2pt)[
        #box(width: track * val / max-val, height: bar-h, fill: col, radius: 2pt)
      ],
      [#text(weight: "bold", size: 9.3pt)[#val] #text(size: 8.5pt, fill: mid-gray)[#note]],
    ))
  )
}

// ---- title ------------------------------------------------------------------
#v(0.5em)
#align(center)[
  #text(size: 19pt, weight: "bold", fill: navy)[How the FPL System Actually Works]
  #v(0.3em)
  #text(size: 12pt, fill: ink)[An illustrated plain-language technical walkthrough]
  #v(0.2em)
  #text(size: 9.5pt, style: "italic", fill: mid-gray)[
    Written for a reader with a business/finance background. No programming assumed.
  ]
  #v(0.4em)
  #line(length: 40%, stroke: 2pt + teal)
]
#v(1em)

= The bird's-eye view

Fantasy Premier League is, structurally, a portfolio management problem. Every week you hold a
portfolio of 15 assets (players), each with a price, and each paying an uncertain weekly
dividend (points). You have a fixed budget, position limits, diversification rules (max 3
players per real club), and transaction costs (transfers beyond the first cost you 4 points).
Your job is to maximize total dividends over the season.

Any investment process like this splits into two distinct jobs, and so does this system:

+ *The forecaster* — the "research department." For every player, every week, it estimates how
  many points he will score in the next few gameweeks. This is a statistical/machine-learning
  problem.
+ *The optimizer* — the "portfolio construction desk." Given those forecasts, it works out the
  best legal squad, which transfers to make, whom to captain, and when to play chips. This is a
  mathematical optimization problem with a known, exact answer once the forecasts are fixed.

The separation matters because the two jobs fail differently. The forecaster's world is noisy
and probabilistic — it can only be "less wrong." The optimizer's world is deterministic — given
inputs, it finds the provably best squad, so if the final squads are bad, the fault is almost
always the forecasts, not the optimization.

#let stage(title, sub, col) = box(
  width: 100%, inset: (x: 5pt, y: 7pt), radius: 4pt,
  fill: col.lighten(88%), stroke: 1pt + col,
)[
  #align(center)[
    #text(size: 9pt, weight: "bold", fill: col.darken(15%))[#title]
    #v(0.15em)
    #text(size: 7.6pt, fill: ink.lighten(20%))[#sub]
  ]
]
#let arrow = align(center + horizon)[#text(size: 13pt, fill: mid-gray)[#sym.arrow.r]]

#figure(
  block(width: 100%, inset: (y: 4pt))[
    #grid(
      columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
      column-gutter: 4pt,
      row-gutter: 6pt,
      stage("Raw data", [6 seasons, \~163k player-weeks], mid-gray),
      arrow,
      stage("Features", [\~115 predictors per row], navy),
      arrow,
      stage("Models", [one per position, chosen by bake-off], navy),
      arrow,
      stage("Forecasts", [points per player per week], navy),
      arrow,
      stage("Optimizer", [MILP: exact best squad], teal),
      arrow,
      stage("Decisions", [squad, transfers, captain, chips], teal),
    )
    #v(4pt)
    #align(center)[
      #text(size: 8pt, fill: mid-gray)[
        #box(width: 8pt, height: 8pt, fill: navy.lighten(88%), stroke: 1pt + navy) forecaster
        ("research") #h(1.2em)
        #box(width: 8pt, height: 8pt, fill: teal.lighten(88%), stroke: 1pt + teal) optimizer
        ("portfolio desk")
      ]
    ]
  ],
  caption: [The pipeline. Everything left of the optimizer is statistics; everything from the
    optimizer on is exact mathematics.],
)

= The data

The raw material is one row per *player per gameweek*: minutes played, goals, assists, bonus
points, expected goals (xG — a bookmaker-style estimate of how many goals a player "should"
have scored given the quality of his shots), the opponent, home/away, price, ownership, and so
on. Six seasons (2020-21 through 2025-26) give roughly *163,000 rows*.

Two sourcing details worth knowing:

- Historical data comes from a well-maintained public archive of FPL statistics; live data —
  next week's fixtures, your current squad — comes from FPL's official API, since the archive
  has no rows for matches not yet played.
- Gameweeks are numbered on one continuous counter across all seasons (GW 1-228 so far) rather
  than resetting each August. This makes "train on everything before week $t$" trivial to
  express — which, as you'll see, is the single most important discipline in the whole system.

= Feature engineering: turning history into predictors

A model cannot eat a player's biography; it eats numbers. A *feature* is one number per
player-week that summarizes something predictive. This system builds \~115 of them per row:

- *Form* — rolling averages of points, minutes, xG etc. over the last 3 and 5 games, plus a
  season-to-date and a career-to-date average. Like the 1-month, 3-month, YTD and
  since-inception returns on a fund factsheet: same quantity, different memory lengths.
- *Exponentially-weighted form* — recent games count more (half-life of 3 gameweeks). The
  finance analogue is exponentially-weighted volatility (RiskMetrics) versus a simple window.
- *Efficiency rates* — goals and xG per 90 minutes rather than per game, so a striker who
  scored in a 20-minute cameo isn't confused with one who needed the full match.
- *"Nailedness"* — the share of recent games started and played 60+ minutes. A brilliant player
  who doesn't get on the pitch pays no dividend.
- *Opponent strength* — the upcoming opponent's rolling attacking and defensive form, computed
  at team level; what separates "Haaland at home to the bottom club" from "Haaland away at
  Arsenal."
- *Fixture difficulty* — FPL's published 1-5 difficulty rating for the coming fixtures.
- *FPL's own forecast (xP)* — FPL publishes its own expected-points number pre-match; we feed
  in its recent history, like including consensus analyst estimates in your own equity model.

== The one rule everything depends on: no peeking

Every feature derived from a player's own performance is *shifted back one gameweek*: the row
for gameweek $t$ contains only information from $t-1$ and earlier. The forecast for next
Saturday cannot contain anything from next Saturday. This is the modeling equivalent of
*look-ahead bias* in a trading backtest — a strategy that "knew" tomorrow's close looks
brilliant in-sample and dies in production — and it creeps in through cracks, not the front
door. Two real bugs found and fixed in this project's internal review:

- The live weekly forecast was accidentally reusing each player's *last played* match row, so
  its "fixture difficulty" described last week's opponent, not next week's.
- The model-blending weights (@sec-combine) were estimated on the very window the backtest was
  scored on — a subtle self-grading loop worth roughly *100 phantom points* (@sec-results).

One more honesty rule: statistics that didn't exist in early seasons (xG wasn't collected
before 2022-23) are stored as *missing*, not zero. "We didn't measure it" and "he generated
exactly none" are different facts, and the models treat them differently.

= The forecasting models

== Why four models, not one

The system trains a separate model per position (GK, DEF, MID, FWD). What predicts a
goalkeeper's points (saves, clean sheets) has almost nothing in common with what predicts a
forward's (xG, penalty duty). Pooling them forces one set of rules to serve four different
games.

== The registry: twelve candidates and an index to beat

Rather than betting on one algorithm, the project maintains a *registry* of twelve — from plain
linear regression up through modern gradient-boosted trees (LightGBM, XGBoost, CatBoost),
random forests, nearest-neighbour and support-vector methods. Every candidate is trained and
scored identically, and results are reported even when a method flops: the research log reads
like a lab notebook, not a highlight reel.

One member has special status: *ordinary least squares (OLS) regression is the designated
index*. Exactly like a passive benchmark in asset management, every fancier method must beat
plain OLS on held-out data to justify its complexity. (Several don't.)

#figure(
  block(width: 100%)[
    #v(2pt)
    #hbars(
      (
        ("naive: last week again", 1.00, mid-gray, "= 1.0 by definition"),
        ("OLS regression (the index)", 0.84, navy.lighten(35%), ""),
        ("CatBoost (production)", 0.70, teal, ""),
      ),
      1.05,
    )
    #v(2pt)
    #align(center)[#text(size: 8.5pt, fill: mid-gray)[MASE, averaged across positions on held-out weeks — lower is better; below 1.0 beats the naive rule]]
  ],
  caption: [What "forecast skill" looks like here. The naive "he'll score what he scored last
    week" rule defines 1.0; the index beats it comfortably; the production model beats the
    index. Fantasy points are so noisy that even the best model only removes about 30% of the
    naive error.],
)

== What a gradient-boosted tree is, in one paragraph

The workhorse models here are *gradient-boosted decision trees*. A decision tree is a flowchart
of yes/no questions ("has he averaged over 4 points in the last 5 games? is the opponent's
defensive form weak?") ending in a numeric prediction. One tree is crude. Boosting builds
hundreds of small trees *in sequence, each fit to the errors the previous ones still make* — a
chain of specialists, each correcting the last. For tabular data like ours, this family has
dominated practical machine learning for a decade — which is why it displaced the neural
network (LSTM) this project started with: sequence models need far more data per player than 38
gameweeks a season provides, while boosted trees pool learning across all players.

== The median-versus-mean trap <sec-trap>

Models are trained by minimizing a *loss function*, and that choice quietly decides *what kind
of guess* the model makes: squared-error loss produces the conditional *mean*, absolute-error
(MAE) loss produces the conditional *median*. For symmetric data those coincide. FPL points are
wildly asymmetric — most weeks a player returns 2 points, occasionally he explodes for 15 —
like income distributions, where a few enormous values pull the mean far above the median.

#figure(
  block(width: 100%)[
    #let W = 13.2cm
    #let H = 3.1cm
    #let BAND = 30pt   // reserved label strip above the curve so nothing overlaps the peak
    #let f(x) = { calc.exp(-calc.pow(calc.ln(calc.max(x, 0.05)) - 1.0, 2.0) / 0.9) / calc.max(x, 0.05) }
    #let xmax = 13.0
    #let n = 90
    #let fmax = 0.38
    #let pts = range(0, n + 1).map(i => {
      let x = i / n * xmax
      (x / xmax * W, BAND + H - calc.min(f(x) / fmax, 1.0) * (H - 4pt))
    })
    #let xpos(x) = x / xmax * W
    #box(width: W, height: BAND + H + 20pt)[
      // density silhouette
      #place(top + left, polygon(
        fill: navy.lighten(82%), stroke: 1.1pt + navy.lighten(30%),
        (0pt, BAND + H), ..pts, (W, BAND + H),
      ))
      // baseline
      #place(top + left, line(start: (0pt, BAND + H), end: (W, BAND + H), stroke: 0.8pt + ink))
      // median line + label (label sits in the reserved band, right-anchored left of the line)
      #place(top + left, dx: xpos(2.6), line(start: (0pt, BAND - 4pt), end: (0pt, BAND + H), stroke: (paint: teal, thickness: 1.4pt, dash: "dashed")))
      #place(top + left, dy: 0pt)[#box(width: xpos(2.6) - 4pt)[#align(right)[#text(size: 8.2pt, hyphenate: false, fill: teal.darken(10%), weight: "bold")[median ≈ 2\ #text(weight: "regular")[what MAE loss predicts]]]]]
      // mean line + label (left-anchored right of the line)
      #place(top + left, dx: xpos(4.7), line(start: (0pt, BAND - 4pt), end: (0pt, BAND + H), stroke: (paint: brick, thickness: 1.4pt, dash: "dashed")))
      #place(top + left, dx: xpos(4.7) + 4pt, dy: 0pt)[#box(width: W - xpos(4.7) - 4pt)[#align(left)[#text(size: 8.2pt, fill: brick, weight: "bold")[mean ≈ 4\ #text(weight: "regular")[what the optimizer needs]]]]]
      // tail annotation
      #place(top + left, dx: xpos(8.4), dy: BAND + H - 40pt)[#text(size: 8.2pt, fill: amber.darken(15%), weight: "bold")[haul weeks: rare, huge,\ #text(weight: "regular")[and what captaincy is about]]]
      // x axis label
      #place(top + left, dy: BAND + H + 3pt)[#box(width: W)[#align(center)[#text(size: 8.5pt, fill: mid-gray)[points scored in one gameweek #sym.arrow.r]]]]
    ]
  ],
  caption: [A right-skewed points distribution (illustrative). One model cannot serve three
    masters: accuracy metrics reward the median, squad value depends on the mean, captaincy
    depends on the tail.],
) <fig-skew>

Our best forecaster, CatBoost trained with MAE loss, predicts something close to each player's
*median* week — its forecasts sum to only about half the points actually scored league-wide. Is
that bad? It depends what the number is used for, and this became a central finding:

- For *ranking* players ("who is better this week?"), median-style forecasts are excellent —
  this model identifies the week's top scorer better than any alternative tested.
- For *absolute-scale decisions* (is a transfer worth its 4-point fee?), a forecast at half the
  true level distorts the exchange rate between points and fees.

We tested the obvious fix — rescaling each position's forecasts so they sum correctly,
estimated on past data only — and it made the squads *worse* (@sec-results). A theoretically
motivated correction is a hypothesis, not a fix, until the backtest agrees.

== Combining models: a diversification story that failed, instructively <sec-combine>

With twelve forecasters, the natural instinct is to blend them — diversification for models.
The system estimated optimal blend weights, and for a long time the blend was the production
forecaster. Then a clean head-to-head showed the *single best model beat the twelve-model blend
at every position*.

#keyidea("The Markowitz parallel")[
  Optimized portfolio weights, estimated from finite noisy data, routinely lose out-of-sample
  to naive equal weighting — estimation error in the weights swamps the theoretical gain from
  optimizing. The same mathematics bites here (Clemen 1989, in the forecasting literature):
  twelve highly correlated forecasters, weights estimated on a short window — the weights were
  fitting noise. We tested the standard remedies too (equal-weighting the top 3, ridge-shrunk
  weights); the single best model still won.
]

So production is *one CatBoost model per position* — chosen not by ideology but by a *bake-off*
re-run on every training pass: single-best vs. three combination schemes, all scored on the
same held-out weeks. If a future change ever makes a blend win again, production follows the
evidence automatically.

= How we know whether any of this works

== Walk-forward evaluation: the honest backtest

All evaluation mimics real life: to forecast gameweek $t$, train only on gameweeks strictly
before $t$; predict; roll forward; repeat. No model, blend weight, scaling factor or tuning
parameter is ever estimated on data from the period it is judged on — exactly how a quant fund
should backtest a strategy, for exactly the same reason.

#figure(
  block(width: 100%)[
    #let cell(fill, stroke) = box(width: 11pt, height: 11pt, fill: fill, stroke: stroke, radius: 1.5pt)
    #let train = cell(navy.lighten(72%), 0.7pt + navy.lighten(40%))
    #let test = cell(teal.lighten(20%), 0.9pt + teal.darken(10%))
    #let future = cell(white, 0.7pt + mid-gray.lighten(30%))
    #let row(k, total) = {
      let cells = ()
      for i in range(0, total) {
        if i < k { cells.push(train) } else if i == k { cells.push(test) } else { cells.push(future) }
      }
      stack(dir: ltr, spacing: 2.5pt, ..cells)
    }
    #set text(size: 8.8pt)
    #grid(
      columns: (auto, auto),
      column-gutter: 12pt, row-gutter: 5pt,
      align: (right + horizon, left + horizon),
      [step 1], row(6, 14),
      [step 2], row(7, 14),
      [step 3], row(8, 14),
      [step 4], row(9, 14),
      [#text(fill: mid-gray)[⋮]], [#text(fill: mid-gray)[⋮ #h(2pt) one step per gameweek, to the end of the data]],
    )
    #v(4pt)
    #align(center)[#text(size: 8.5pt, fill: mid-gray)[
      #box(width: 8pt, height: 8pt, fill: navy.lighten(72%), stroke: 0.7pt + navy.lighten(40%)) train (past)
      #h(1.2em) #box(width: 8pt, height: 8pt, fill: teal.lighten(20%), stroke: 0.9pt + teal.darken(10%)) predict (this week)
      #h(1.2em) #box(width: 8pt, height: 8pt, stroke: 0.7pt + mid-gray.lighten(30%)) unseen future
    ]]
  ],
  caption: [Walk-forward validation: the training window only ever grows into the past; the
    week being predicted is never part of it. Each square is a gameweek.],
)

== The scorecards

- *MAE* — on average, how many points off is each player-week forecast? (\~0.85-1.05 here.)
- *MASE* — MAE divided by the error of a naive "same as last week" forecast; below 1.0 beats
  naive. The headline accuracy metric, because raw MAE is uninterpretable for a bursty series.
- *Decision-aligned diagnostics* — calibration (do forecasts sum to points actually scored?),
  within-week rank correlation, and *top-1 capture* (how often is our predicted best player
  actually the week's best — the captaincy question). These exist because of @sec-trap: a
  model can improve MASE while becoming worse for decisions.
- *The gold standard: realized points* — feed the forecasts through the optimizer over a full
  historical season and count what the squads actually scored. The only number that measures
  the whole system.

That last one earns its title because — demonstrated twice in this project — *forecast accuracy
improvements do not reliably produce better squads*. A feature upgrade once improved MASE at
every position while realized points dropped 1,966 → 1,880. Accuracy metrics are the compass;
realized points are the terrain.

= The optimizer

Given predicted points for every player over the next few weeks, squad selection becomes a
*mixed-integer linear program* (MILP) — "integer" because you cannot hold 0.4 of a defender.
It is the same mathematical species as portfolio optimization with lot-size constraints, and it
is solved *exactly*: maximize predicted points subject to budget (£100m), squad shape
(2 GK / 5 DEF / 5 MID / 3 FWD), max 3 per club, a legal starting XI, captain scoring double,
transfer accounting (one free per week, extras cost 4 points), and one-time chips.

Because optimizing a whole season at once is computationally hopeless (and pointless — forecasts
degrade with distance), it solves on a *rolling horizon*: optimize weeks $t$ to $t+2$, commit
only week $t$'s decisions, roll forward, re-solve. Corporate planning under uncertainty works
the same way. An automated test suite checks every output against all the rules above, so a
code change that quietly produces illegal squads is caught by tests, not by a mysterious
backtest number.

= What the numbers actually say <sec-results>

All figures are realized points over the same 31-gameweek window (2024-25 season, weeks 1-31),
same optimizer, directly comparable:

#figure(
  block(width: 100%)[
    #v(2pt)
    #hbars(
      (
        ("thesis-era LSTM", 1526, mid-gray.lighten(20%), ""),
        ("first Python rewrite (LightGBM)", 1811, navy.lighten(45%), ""),
        ("model blend, longer history", 1966, amber.lighten(25%), "≈100 pts leakage-inflated"),
        ("honest re-run: 12-model blend", 1869, navy.lighten(25%), "the clean untuned baseline"),
        ("honest re-run: CatBoost only", 1856, navy.lighten(25%), "tie, at 1/12th the cost"),
        ("CatBoost + level rescaling", 1800, brick.lighten(25%), "the 'obvious fix' that failed"),
        ("CatBoost, tuned (Optuna)", 2107, teal, "current best — +13.5%"),
      ),
      2200,
    )
    #v(2pt)
    #align(center)[#text(size: 8.5pt, fill: mid-gray)[realized points, GW1-31 of 2024-25, identical optimizer and rules]]
  ],
  caption: [The full lineage. Honest validation first *lowered* the headline (the amber bar was
    partly phantom), then hyperparameter tuning delivered the largest genuine gain in the
    project's history.],
)

Three takeaways worth internalizing, because they generalize far beyond fantasy football:

+ *Most of the improvement came from process, not cleverness* — honest validation, leakage
  fixes, longer history, automated tuning, and letting a bake-off (not a preference) choose the
  model. The glamorous additions (37 new features in one batch; model blending; level
  recalibration) added roughly nothing or less than nothing. The single largest gain — tuning,
  +251 points — came from patiently searching configuration knobs, and its winning settings
  tell their own story: slower learning rates and shallower trees than the hand-set defaults,
  i.e. the defaults were overconfident.
+ *The metric you optimize is a design decision with consequences.* MAE-trained models predict
  medians; optimizers consume absolute scales; captaincy consumes tail behaviour (@fig-skew).
  No single number captures all three, which is why the system reports a dashboard, not one
  score.
+ *Negative results are assets.* The failed recalibration, the failed blend, the
  accuracy-up/points-down episodes — each is logged with numbers and mechanism in the research
  log, and each permanently changed how the next experiment is judged.

= Where the project goes from here

In priority order — all subject to the same rule: beat 2,107 realized points honestly, or be
reported as a negative result.

- *Consolidate the tuning win*: retrain production on the tuned parameters, tune the other
  boosted models so the registry comparison is fair (a tuned blend deserves a re-match), and
  stability-check the 2,107 on a second window and seed.
- *Upside-aware captaincy*: a parallel quantile model already estimates each player's
  90th-percentile week; feeding that into the captain choice attacks the tail directly.
- *Component decomposition* (the big architectural bet): predict points' ingredients — minutes,
  goals, assists, clean sheets — and recombine. Clean sheets are a team-level event, so eleven
  defenders' forecasts could share one team model.
- *A live-season interface*: the weekly driver already refreshes data, retrains, and prints
  recommended transfers via the FPL API; a friendlier surface is future work once the 2026-27
  season opens.

#v(1em)
#line(length: 100%, stroke: 0.5pt + mid-gray)
#text(size: 9pt, style: "italic", fill: mid-gray)[
  Companion documents: `report/main.typ` (formal write-up, full methodology) ·
  `RESEARCH_LOG.md` (chronological lab notebook, including everything that failed) ·
  `TODO.md` (open items) · `CLAUDE.md` (architecture reference).
]
