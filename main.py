import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from job_filtering import fetch_and_filter_jobs
from llm_models import LLMModels
from modify_resume import modify_resume
from tasks import Tasks

load_dotenv(override=True)

if __name__ == "__main__":
    args = ArgumentParser()

    args.add_argument(
        "--task",
        type=str,
        default=Tasks.FILTER_JOBS.value,
        choices=[task.value for task in Tasks],
        help="Task to perform",
    )

    args.add_argument(
        "--model",
        type=str,
        default=LLMModels.GEMINI.value,
        choices=[model.value for model in LLMModels],
        help="AI model to use",
    )

    args.add_argument(
        "--curr_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Current directory for outputs",
    )
    parsed_args = args.parse_args()

    os.makedirs(os.path.join(parsed_args.curr_dir, "outputs"), exist_ok=True)

    if parsed_args.task == Tasks.FILTER_JOBS.value:
        fetch_and_filter_jobs(parsed_args)
    elif parsed_args.task == Tasks.MODIFY_RESUME.value:
        modify_resume(parsed_args)
