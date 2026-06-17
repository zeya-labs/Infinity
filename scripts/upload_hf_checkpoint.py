from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a checkpoint file or folder to a Hugging Face Hub repository.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository, for example username/repo-name or org/repo-name.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Local checkpoint file or directory to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Remote path inside the repository. Defaults to checkpoints/<checkpoint name>.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hugging Face repository type.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Target branch or revision. Defaults to the Hub repository default branch.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message. Defaults to a checkpoint upload message.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the target repository if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When creating a repository, make it private.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned upload without contacting Hugging Face.",
    )
    return parser


def default_path_in_repo(checkpoint: Path) -> str:
    return f"checkpoints/{checkpoint.name}"


def upload_checkpoint(args: argparse.Namespace) -> str | None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint}: checkpoint path does not exist")

    path_in_repo = args.path_in_repo or default_path_in_repo(checkpoint)
    commit_message = args.commit_message or f"Upload {checkpoint.name}"

    print(f"repo_id: {args.repo_id}")
    print(f"repo_type: {args.repo_type}")
    print(f"checkpoint: {checkpoint}")
    print(f"path_in_repo: {path_in_repo}")
    if args.revision:
        print(f"revision: {args.revision}")
    if args.create_repo:
        visibility = "private" if args.private else "public"
        print(f"create_repo: yes ({visibility})")

    if args.dry_run:
        print("dry_run: no upload performed")
        return None

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `python -m pip install huggingface_hub` "
            "or reinstall requirements.txt."
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    common_kwargs = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "commit_message": commit_message,
    }
    if checkpoint.is_dir():
        commit_info = api.upload_folder(
            folder_path=checkpoint,
            path_in_repo=path_in_repo,
            **common_kwargs,
        )
    else:
        commit_info = api.upload_file(
            path_or_fileobj=checkpoint,
            path_in_repo=path_in_repo,
            **common_kwargs,
        )

    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url:
        print(f"uploaded: {commit_url}")
    else:
        print("uploaded")
    return commit_url


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    upload_checkpoint(args)


if __name__ == "__main__":
    main()
