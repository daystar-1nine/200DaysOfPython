# ==============================================================================
# Program    : CLI Study Tracker Tool (Bonus Challenge)
# Objective  : Polish CLI study tracker application supporting add, list, complete, and summary.
# Concept    : CLI Subcommands, .env Configuration, JSON Persistence & Logging
# Why Used   : Combines argparse subparsers, environment variable fallback, and JSON storage.
# ==============================================================================

import argparse
import json
import logging
import os
import sys

DATA_FILE = os.path.join(os.path.dirname(__file__), "study_tasks.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "study_cli.log")

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def load_config() -> dict:
    """Loads configuration from environment variables or defaults."""
    app_name = os.getenv("APP_NAME", "StudyTrackerCLI")
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    return {"app_name": app_name, "debug": debug_mode}

def load_tasks() -> list[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed loading study tasks: %s", e, exc_info=True)
        return []

def save_tasks(tasks: list[dict]) -> None:
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=4)
    except Exception as e:
        logging.error("Failed saving study tasks: %s", e, exc_info=True)

def handle_add(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    next_id = max([t.get("id", 0) for t in tasks], default=0) + 1
    new_task = {
        "id": next_id,
        "topic": args.topic,
        "hours": args.hours,
        "completed": False
    }
    tasks.append(new_task)
    save_tasks(tasks)
    logging.info("Added Study Task ID %d: '%s' (%.1f hrs)", next_id, args.topic, args.hours)
    print(f"[SUCCESS] Added Study Task #{next_id}: '{args.topic}' ({args.hours} hrs)")

def handle_list(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    if not tasks:
        print("No study tasks found.")
        return
    print("\n------------------ STUDY TASKS LIST ------------------")
    for t in tasks:
        status = "[DONE]" if t["completed"] else "[PENDING]"
        print(f"ID: {t['id']:<3} | Status: {status:<9} | Topic: {t['topic']:<25} | Hours: {t['hours']} hrs")
    print("------------------------------------------------------\n")

def handle_complete(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    found = False
    for t in tasks:
        if t["id"] == args.id:
            t["completed"] = True
            found = True
            break
    if found:
        save_tasks(tasks)
        logging.info("Completed Study Task ID %d", args.id)
        print(f"[SUCCESS] Marked Study Task #{args.id} as COMPLETED!")
    else:
        logging.warning("Complete failed: Task ID %d not found", args.id)
        print(f"[WARNING] Study Task #{args.id} not found.")

def handle_summary(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    if not tasks:
        print("No study tasks found.")
        return
    total_hours = sum(t["hours"] for t in tasks)
    completed_hours = sum(t["hours"] for t in tasks if t["completed"])
    completed_count = sum(1 for t in tasks if t["completed"])

    print("\n------ STUDY PROGRESS SUMMARY ------")
    print(f"Total Tasks     : {len(tasks)}")
    print(f"Completed Tasks : {completed_count}")
    print(f"Total Hours     : {total_hours:.1f} hrs")
    print(f"Completed Hours : {completed_hours:.1f} hrs")
    print("------------------------------------\n")

def create_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description=f"{config['app_name']} - Personal Study Tracker")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Add Command
    add_parser = subparsers.add_parser("add", help="Add a new study topic")
    add_parser.add_argument("topic", type=str, help="Study topic title")
    add_parser.add_argument("--hours", type=float, default=1.0, help="Estimated study hours")

    # List Command
    list_parser = subparsers.add_parser("list", help="List all study tasks")

    # Complete Command
    complete_parser = subparsers.add_parser("complete", help="Mark a study task as complete")
    complete_parser.add_argument("id", type=int, help="Task ID to mark complete")

    # Summary Command
    summary_parser = subparsers.add_parser("summary", help="Display summary statistics")

    return parser

def main() -> None:
    print("=== BONUS CHALLENGE: CLI STUDY TRACKER TOOL ===")
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        # 1. Add "Python Concurrency"
        handle_add(parser.parse_args(["add", "Python Concurrency", "--hours", "2.5"]))
        # 2. Add "CLI Applications"
        handle_add(parser.parse_args(["add", "CLI Applications", "--hours", "1.5"]))
        # 3. List
        handle_list(parser.parse_args(["list"]))
        # 4. Complete ID 1
        handle_complete(parser.parse_args(["complete", "1"]))
        # 5. Summary
        handle_summary(parser.parse_args(["summary"]))
    else:
        args = parser.parse_args()
        if args.command == "add":
            handle_add(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "complete":
            handle_complete(args)
        elif args.command == "summary":
            handle_summary(args)

if __name__ == "__main__":
    main()
