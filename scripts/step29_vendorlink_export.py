"""One fixed export; private qrels never enter qualification or result logs."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

from step29_vendorlink_data import Group, build_views
from step29_vendorlink_run import file_sha256, write_json


def export(rows, partition, output, custody, expected):
    if output.exists() or custody.exists():
        raise ValueError("Fresh output and private custody required")
    payloads, labels, report = build_views(rows, **partition)
    counts = {}
    for key, values in payloads.items():
        groups = [Group.from_dict(v) for v in values]
        sizes = [len(g.views)//2 for g in groups]
        counts[key] = dict(accounts=sum(sizes), groups=len(groups), views=2*sum(sizes),
                           selected_ads=sum(len(v.records) for g in groups for v in g.views),
                           group_account_sizes={str(k): v for k, v in Counter(sizes).items()})
    if set(counts) != set(expected):
        raise ValueError("Export partition set differs from the fixed preview")
    for key, value in counts.items():
        reference = dict(expected[key])
        reference["group_account_sizes"] = {str(k): v for k, v in reference["group_account_sizes"].items()}
        if any(value[k] != reference[k] for k in value):
            raise ValueError("Export counts differ: "+key)
    output.mkdir(parents=True)
    custody.mkdir(parents=True, mode=0o700)
    manifest, samples = [], {}
    for key, values in sorted(payloads.items()):
        domain, split = key.split(":")
        path = output/f"{domain}_{split}.json"
        write_json(path, values)
        manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path)))
        if split == "fit":
            classes = sorted({v["identity"] for g in values for v in g["views"]})
            # Fixed source order among maximum-size groups; no model outcome selection.
            chosen = [g for g in values if len(g["views"]) == 16][:3]
            if len(chosen) != 3:
                raise ValueError("Three full fit groups required for this qualification")
            small = output/f"{domain}_qualification.json"
            write_json(small, dict(domain=domain, classes=classes, groups=chosen))
            samples[domain] = dict(path=str(small), bytes=small.stat().st_size, sha256=file_sha256(small),
                                   classes=len(classes), groups=3)
        else:
            # Only preparation sees labels; inference is given the already blinded files.
            private = custody/f"{domain}_{split}_qrels.json"
            write_json(private, {g["uid"]: dict(view_ids=[v["uid"] for v in g["views"]],
                                                labels=labels[g["uid"]]) for g in values})
            private.chmod(0o600)
            manifest.append(dict(path=str(private), bytes=private.stat().st_size, sha256=file_sha256(private)))
    result = dict(status="FIXED_VIEWS_EXPORTED_LABELS_SEALED", counts=counts, files=manifest,
                  qualification_samples=samples, report=report, evaluation_opened=False,
                  shared_exact_text_unit="account--masked-text-fingerprint objects")
    write_json(output/"manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    permit = json.loads(args.permit.read_text())
    if (sys.platform != "linux" or permit.get("operation") != "vendorlink_native_qualification"
            or not permit.get("approved") or not permit.get("implementation_review_closed")
            or permit.get("config_sha256") != file_sha256(args.config)):
        raise ValueError("Approved fixed config and closed implementation review required")
    source = Path(config["canonical"]["path"])
    if source.stat().st_size != config["canonical"]["bytes"] or file_sha256(source) != config["canonical"]["sha256"]:
        raise ValueError("Canonical source differs")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    result = export(rows, config["partition"], Path(config["export"]), Path(config["custody"]),
                    config["expected_partitions"])
    print(json.dumps(dict(status=result["status"], partitions=len(result["counts"])), flush=True))


if __name__ == "__main__":
    main()
