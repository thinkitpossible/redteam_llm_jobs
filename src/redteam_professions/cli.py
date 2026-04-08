from __future__ import annotations

import argparse
import json
import sys

from .job_spec import job_dict_to_arg_defaults, load_job_file
from .pipeline import PipelineOptions, RedTeamBatchPipeline, parse_line_kinds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate profession-focused red-team prompts.")
    parser.add_argument(
        "--job",
        default=None,
        metavar="PATH",
        help="JSON job file: fields become argparse defaults (CLI flags override).",
    )
    parser.add_argument("--input", dest="input_path", help="Path to the profession .docx file.")
    parser.add_argument("--output-dir", default="output", help="Directory for exported artifacts.")
    parser.add_argument("--limit", type=int, help="Only process the first N professions after filtering.")
    parser.add_argument(
        "--profession-id",
        action="append",
        default=[],
        help="Specific profession_id to process. Repeatable.",
    )
    parser.add_argument(
        "--profession-contains",
        action="append",
        default=[],
        help="Only process professions whose normalized name contains this keyword. Repeatable.",
    )
    parser.add_argument(
        "--line-kind",
        action="append",
        default=[],
        dest="line_kind",
        metavar="KIND",
        help="Only generate selected lines: active (从业者主动线), passive (外部针对从业者). Repeatable.",
    )
    parser.add_argument(
        "--generation-round",
        type=int,
        default=1,
        help="Tag output samples with a generation round number.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force regeneration even if per-profession status already exists.",
    )
    parser.add_argument(
        "--no-bundles",
        action="store_true",
        help="Skip writing per-profession bundle.json (dual-branch aggregate).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Required: generate via packaged active/passive prompts under prompts/ (OpenAI-compatible Chat + REDTEAM_LLM_* env).",
    )
    parser.add_argument(
        "--llm-min-response-len",
        type=int,
        default=None,
        metavar="N",
        help="Min length of parsed user_utterance to accept (overrides env REDTEAM_LLM_MIN_RESPONSE_LEN).",
    )
    parser.add_argument(
        "--llm-min-hooks-response-len",
        type=int,
        default=None,
        metavar="N",
        help="Min raw LLM response length for legal-hooks JSON (overrides env REDTEAM_LLM_HOOKS_MIN_RESPONSE_LEN).",
    )
    parser.add_argument(
        "--enable-judgement",
        action="store_true",
        help="Run relevance/dedup judge and split rejected samples (default: off, all samples accepted).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--job", default=None)
    peek_args, _ = peek.parse_known_args(argv_list)
    job_defaults: dict = {}
    if peek_args.job:
        job_defaults = job_dict_to_arg_defaults(load_job_file(peek_args.job))

    parser = build_parser()
    parser.set_defaults(**job_defaults)
    args = parser.parse_args(argv_list)
    try:
        options = PipelineOptions(
            input_path=args.input_path,
            output_dir=args.output_dir,
            limit=args.limit,
            profession_ids=set(args.profession_id),
            profession_contains=list(args.profession_contains),
            line_kinds=parse_line_kinds(args.line_kind),
            rerun=args.rerun,
            generation_round=args.generation_round,
            write_bundles=not args.no_bundles,
            use_llm=args.use_llm,
            llm_min_response_len=args.llm_min_response_len,
            llm_min_hooks_response_len=args.llm_min_hooks_response_len,
            enable_judgement=args.enable_judgement,
        )
        llm_client = None
        if args.use_llm:
            from .llm.client import OpenAICompatibleChatClient
            from .llm.config import LlmConfig

            llm_client = OpenAICompatibleChatClient(LlmConfig.from_env())
        manifest = RedTeamBatchPipeline(llm_client=llm_client).run(options)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
