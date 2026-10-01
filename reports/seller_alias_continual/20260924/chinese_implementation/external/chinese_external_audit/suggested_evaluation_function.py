# Reviewer proposal; not merged into submitted source.
def fixed_zero_summary(counts: list[dict], domains: list[str]) -> dict:
    """Aggregate saved group counts; no labels and no new threshold selection."""
    values = np.asarray([[row[key] for key in ("tp", "fp", "fn", "tn")]
                         for row in counts], dtype=np.int64)
    domain_array = np.asarray(domains)
    return {"threshold": 0.0,
            "by_domain": {d: rates(values[domain_array == d]) for d in "ABC"},
            "pooled": rates(values)}


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != "Linux" or destination.exists():
        raise ValueError("Requires resumed Linux evaluation and a new directory")
    c = contract()
    groups, metadata, checked = public.public_inputs(c)
    arrays, part, r = validated_run(out, c, groups, metadata, checked)
    reference = historical_reference(c, part)
    destination.mkdir(parents=True)
    data.write_json(destination / "access.json", {"status": "ALL_FOUR_MODELS_COMPLETE_BEFORE_VALID",
                    "development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = public.attach_labels(groups["development"], c, "development")
        truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
        e = c["evaluation"]
        domains = [row["domain"] for row in part["development"]]
        result = {"status": "CHINESE_BASE_DEVELOPMENT_EVALUATED_NO_AUTOMATIC_WINNER", "config": c,
                  "manifest_sha256": data.sha256(out / "manifest.json"), "columns": list(metrics.COLUMNS),
                  "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0},
                  "arms": {}, "comparisons": {}, "primary_comparison": c["primary_comparison"],
                  "historical_reference": reference["files"],
                  "formal_training_seconds": r["formal_training_seconds"]}
        matrices, auto_counts = {}, {}
        # Phase 1: collect the nine metric matrices and all group confusion counts.
        # Do not invoke bootstrap generation, reports or comparisons in this phase.
        for arm in ARMS:
            (destination / arm).mkdir()
            points = {}
            for epoch in ("3", "6"):
                scores = arrays[arm]["points"][epoch]["development"]
                matrix, counts = metrics.group_metrics(truth, scores)
                path = destination / arm / f"epoch{epoch}_metrics.npy"
                np.save(path, matrix, allow_pickle=False)
                points[epoch] = {"file": data.record(path, destination), "counts_at_logit_zero": counts,
                                 "mean": dict(zip(metrics.COLUMNS, matrix.mean(0).tolist())),
                                 "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist())) for d in "ABC"}}
                if epoch == "6":
                    matrices[arm] = matrix
                    count = binary_counts(truth, scores, arrays[arm]["threshold"])
                    auto_counts[arm] = count
            result["arms"][arm] = {"points": points, "automatic_classification": {
                "threshold": arrays[arm]["threshold"], "counts_by_group": auto_counts[arm].tolist()}}
        matrices["historical_labse"], historical_counts = metrics.group_metrics(truth, reference["scores"])
        auto_counts["historical_labse"] = binary_counts(truth, reference["scores"], reference["threshold"])
        path = destination / "historical_labse_metrics.npy"
        np.save(path, matrices["historical_labse"], allow_pickle=False)
        result["historical_labse"] = {"file": data.record(path, destination),
            "counts_at_logit_zero": historical_counts,
            "threshold": reference["threshold"], "counts_by_group": auto_counts["historical_labse"].tolist()}
        data.write_json(destination / "collected.json", {
            **result, "status": "CHINESE_BASE_METRICS_COLLECTED_NO_BOOTSTRAP", "domains": domains})
        # Phase 2: every quantity below is recoverable from saved matrices/counts/config.
        draws = np.random.default_rng(e["bootstrap_seed"]).integers(0, 20, size=(e["bootstrap_replicates"], 3, 20))
        for arm in ARMS:
            result["arms"][arm]["automatic_classification"].update(automatic_report(auto_counts[arm], domains, draws))
            for point in result["arms"][arm]["points"].values():
                point["fixed_zero_classification"] = fixed_zero_summary(point["counts_at_logit_zero"], domains)
        result["historical_labse"]["automatic_classification"] = automatic_report(auto_counts["historical_labse"], domains, draws)
        result["historical_labse"]["fixed_zero_classification"] = fixed_zero_summary(historical_counts, domains)
        for candidate, baseline in c["comparisons"]:
            result["comparisons"][f"{candidate}_minus_{baseline}"] = {
                "metrics": metric_comparison(matrices[candidate], matrices[baseline], domains, draws),
                "automatic": automatic_comparison(auto_counts[candidate], auto_counts[baseline], domains, draws)}
        result["interpretation"] = "Each domain's fixed-threshold FPR<=0.001 AND recall>=0.5 is required. Point-value development qualification, not independent test, training-seed robustness, calibrated deployment guarantee, or automatic model selection. AP comparisons are secondary."
        data.write_json(destination / "evaluation.json", result)
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {"status": "EVALUATION_FAILED_NO_RETRY",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
