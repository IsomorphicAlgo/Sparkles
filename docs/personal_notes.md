Yes — that kind of skew can make sense with your setup, and it doesn’t by itself mean something is wrong.

Why stop_loss can dominate

Asymmetric barriers — The lower barrier is usually a smaller move (e.g. ~5% vol-scaled) than the upper (e.g. ~15% with a min-profit floor). On 1m data, short dips often tag the stop before price has moved enough to hit take-profit.

Pessimistic same-bar rule — If both could hit on the same bar, stop is checked first, so ties go to stop_loss, which nudges counts in that direction.

Volatility / path — Names like RKLB are jumpy; many entries never “survive” cleanly to TP or vertical without an adverse excursion through the lower band first.

vertical smaller — A lot of paths end at SL or TP before the time horizon, so fewer vertical outcomes is normal.

end_of_data = 1 — Unusual but fine: almost all entries get resolved by a barrier or vertical; one path ran out of history.

What to watch for

If validation is mostly one class, metrics can look misleading — you already saw the “drop val rows with unseen classes” behavior; imbalance is a training/evaluation issue more than a “labels are invalid” issue.
If you want a more balanced label mix for experimentation, that’s a design choice (wider SL, tighter TP, different vertical_max_trading_days, label_entry_stride, etc.) — and would require re-labeling, not just retraining.
So: yes, a much higher stop_loss count can be plausible for asymmetric triple barriers on 1m bars; use it as a signal to check class balance for modeling, not as proof the pipeline is broken.




Here’s a concise way to think about it — nothing here proves the wider stop is “wrong,” but it does change the problem the classifier sees.

1. What actually changed when you moved SL 5% → 10%
For each entry, the lower barrier moves farther down (after the same vol scaling/clamps). So:

Fewer paths hit stop_loss (harder to tag the stop).
More paths can stay “alive” long enough to hit take_profit, hit vertical, or wander until end_of_data.
So you didn’t just “make training easier” — you changed the labels (y) and, in your pipeline, parts of the features too (e.g. sl_move in build_feature_matrix comes from the labeling config). Same ticker, different target and different X.

2. Why accuracy can drop a lot even if the pipeline is fine
Accuracy is very sensitive to how many rows each class has and how separable the classes are from entry-only features.

Easier majority-class game before: With a tight stop, stop_loss was very common. A model can get a decent accuracy by often predicting stop_loss (majority baseline is high).
After widening the stop: That majority shrinks; take_profit / vertical grow. The random / majority baseline for accuracy falls, so both train and val accuracy can fall even if the model is behaving reasonably.
So: lower accuracy does not automatically mean “worse learning” — it often means the task got harder and/or less imbalanced in a way that hurts raw accuracy.

3. Why the task can get harder for your current features
Your model only sees information at the entry bar (plus label-time geometry like sl_move, tp_move_effective, vol, etc.).

When the stop is wider, whether you end up TP vs vertical vs SL depends on more of the future path. That path is not in X. So you’re asking: “From this snapshot, which of several path-dependent outcomes will happen?” — often noisier and less linearly separable for logistic regression.

So it’s plausible that macro-F1 / per-class recall tells a clearer story than accuracy here.

4. What to do to understand (not fix) next
Compare label counts before vs after (sparkles label value counts) — you should see stop_loss down, others up.
Check metrics.json → classification_report_val — which class got worse (precision/recall/F1)?
Think in baselines: e.g. “always predict majority class” accuracy on train/val after the relabel — if that baseline dropped, part of your drop is distribution, not necessarily “model broke.”
Bottom line: Widening stop_loss_base reshapes who wins the triple-barrier race and changes class balance and task difficulty for entry-only features. Falling train and val accuracy is consistent with that — especially if stop_loss stopped dominating. It’s a reason to look at per-class metrics and label counts, not a sign by itself that 10% SL is “wrong.”

Form	Meaning
Omit or null	No weighting (sklearn None).
balanced	Inverse-frequency weights on train labels.
YAML object (dict)	{ "stop_loss": 1.0, "take_profit": 3.0, ... } — keys must match barrier_outcome names; values are weights.

What it’s saying
accuracy = stop_loss recall = 77 / 106 ≈ 0.726 → the model got exactly the 77 true stop_loss rows right and missed all 29 others (7 take_profit, 22 vertical). So on validation it’s behaving like “always predict stop_loss” (or equivalent: it never assigns TP/vertical).
take_profit / vertical: precision and recall 0 → it never predicts those labels correctly (with zero_division=0, that usually means it never predicts those classes at all on val).
stop_loss precision ≈ 0.726 → of all times it said stop_loss, ~72.6% were right — consistent with predicting stop_loss for every row: 77 correct / 106 predicted SL = 0.726.
So this run is a clear majority-class collapse on val, not a subtle mix.

Practical next step
Turn on model.class_weight: balanced (or a dict that up-weights take_profit and vertical), retrain, and run the same PowerShell line again — you’re looking for non-zero recall on TP/vertical even if stop_loss precision drops.

If you use predictions.parquet, a quick check is predicted_class value_counts() — you’ll likely see only stop_loss for this run.