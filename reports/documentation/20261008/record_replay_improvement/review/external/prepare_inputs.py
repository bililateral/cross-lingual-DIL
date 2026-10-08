"""Unpack the already supplied review archive for attachment-only reproduction."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def unpack(path, destination, selected=None):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names = z.namelist() if selected is None else [n for n in z.namelist() if selected(n)]
        z.extractall(destination, members=names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent/'input/review_input.zip')
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    expected = 'f24e5c147561fef9a37cc05842c0843207c95c859b1f076fb47d10f2033fedb7'
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('This review applies to the exact registered input archive only')
    root = args.workspace.resolve()
    if root.exists():
        raise FileExistsError('Choose a new extraction directory; existing work is not overwritten')
    root.mkdir(parents=True)
    (root/'upload').mkdir()
    shutil.copyfile(args.input, root/'upload/review_input(8).zip')
    unpack(args.input, root/'review_current')
    unpack(root/'review_current/background/common_evaluation_input.zip', root/'review_background/common')
    unpack(root/'review_current/background/common_evaluation_review.zip', root/'review_background/review')
    unpack(root/'review_background/common/background/original_result_input.zip', root/'review_background/original')
    unpack(root/'review_background/original/history/implementation_review_input.zip', root/'review_background/implementation',
           lambda n: n.startswith('background/frozen/logit_low_source/') or n in ('manifest.json', 'background/manifest.json'))
    print(json.dumps({'status':'EXTRACTED_ATTACHMENT_ONLY', 'workspace':str(root), 'sha256':expected,
                      'root':str(root/'review_current'), 'background':str(root/'review_background'),
                      'note':'No experiment, model, labels, GPU or project-server operation is performed.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
