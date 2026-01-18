import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from termcolor import cprint

from google_jobs import fetch_google_jobs
from llm_job_filter import MatchedJob, filter_jobs_with_llm

load_dotenv(override=True)

JOB_FILTERS = {
    "yesterday": "since yesterday",
    "last_3_days": "in the last 3 days",
    "full_time": "Full Time",
}


def write_jobs_to_file(filepath, jobs: List[MatchedJob]):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(curr_dir, "templates")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("jobs_report.html")

    html_content = template.render(
        jobs=[job.model_dump() for job in jobs],
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ai_model = "gemini"  # Options: "openai" or "gemini"
    job_titles = [
        "Software Engineer",
        "Machine Learning (ML) Engineer",
        "AI Engineer",
    ]

    cprint("Fetching jobs from Google Jobs...", "cyan")
    all_jobs = []
    for title in job_titles:
        query = " ".join([title, JOB_FILTERS["yesterday"], JOB_FILTERS["full_time"]])
        jobs = fetch_google_jobs(query, pages=1)
        all_jobs.extend(jobs)

    cprint(f"Total jobs fetched: {len(all_jobs)}", "green")
    os.makedirs(os.path.join(curr_dir, "outputs"), exist_ok=True)

    all_jobs_with_scores = []
    for i in range(0, len(all_jobs), 10):
        batch = all_jobs[i : i + 10]
        cprint(f"Processing jobs {i + 1} to {i + len(batch)}", "yellow", end="... ")
        matched_jobs_response = filter_jobs_with_llm(ai_model, batch)
        all_jobs_with_scores.extend(matched_jobs_response.matched_jobs)
        cprint(f"Done ({len(matched_jobs_response.matched_jobs)})", "yellow")

    cprint(f"Total matched jobs: {len(all_jobs_with_scores)}", "green")
    output_filename = f"matched_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    write_jobs_to_file(
        os.path.join(curr_dir, "outputs", output_filename), all_jobs_with_scores
    )
    cprint(f"Matched jobs written to outputs/{output_filename}", "green")


if __name__ == "__main__":
    main()
