from termcolor import cprint

from llm_models import LLMModels
from llm_modify_resume import fetch_updated_resume_details


def modify_resume(parsed_args):
    job_description = ""
    while True:
        cprint(
            "Enter the job description (type 'END' on a new line to finish):", "cyan"
        )
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        job_description = "\n".join(lines)
        if job_description.strip():
            break
        else:
            cprint("Job description cannot be empty. Please try again.", "red")

    resume_details_points = ""
    while True:
        cprint(
            "Enter the resume details points to modify (type 'END' on a new line to finish):",
            "cyan",
        )
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        resume_details_points = "\n".join(lines)
        if resume_details_points.strip():
            break
        else:
            cprint("Resume details points cannot be empty. Please try again.", "red")

    cprint("Modifying resume based on the job description...", "cyan")
    updated_resume = fetch_updated_resume_details(
        LLMModels(parsed_args.model), job_description, resume_details_points
    )
    cprint("Updated Resume:\n", "green")
    cprint(updated_resume)
