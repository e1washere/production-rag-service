"""Seed roadmap issues on GitHub.

Usage:
  export GITHUB_TOKEN=...  # classic repo scope
  python scripts/seed_issues.py --repo e1washere/rag-service
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

ROADMAP: list[dict[str, str]] = [
    {
        "title": "Cloud-ready: Azure App Service deployment",
        "body": (
            "Goals:\n"
            "- Add docker-compose.override.yml for prod\n"
            "- Azure App Service container deploy (Docker)\n"
            "- README: Deploy on Azure button + instructions\n\n"
            "Acceptance Criteria:\n"
            "- App reachable at public URL\n"
            "- Healthcheck passes; logs captured\n"
        ),
        "labels": ["roadmap", "deploy", "azure"],
    },
    {
        "title": "IaC: Terraform for container + storage",
        "body": (
            "Goals:\n"
            "- infra/terraform main.tf for container + storage\n"
            "- README: Infrastructure as Code section\n\n"
            "Acceptance Criteria:\n"
            "- terraform apply provisions app + storage\n"
        ),
        "labels": ["roadmap", "iac", "terraform"],
    },
    {
        "title": "Kubernetes: Helm chart",
        "body": (
            "Goals:\n"
            "- k8s/ deployment.yaml, service.yaml\n"
            "- Helm values.yaml with replicas/resources\n"
            "- README: Deploy to K8s steps\n\n"
            "Acceptance Criteria:\n"
            "- Works on kind/minikube; healthcheck green\n"
        ),
        "labels": ["roadmap", "k8s", "helm"],
    },
    {
        "title": "CI/CD: GitHub Actions deploy to K8s",
        "body": (
            "Goals:\n"
            "- Add job to deploy via kubectl/helm\n"
            "- Trigger on tags\n\n"
            "Acceptance Criteria:\n"
            "- Tag push deploys to cluster\n"
        ),
        "labels": ["roadmap", "ci", "deploy"],
    },
    {
        "title": "Monitoring: Prometheus metrics + Grafana dashboard",
        "body": (
            "Goals:\n"
            "- /metrics endpoint via prometheus_client\n"
            "- Track latency p95, cost/request, HR@k\n"
            "- Provide Grafana dashboard json + README screenshots\n\n"
            "Acceptance Criteria:\n"
            "- Metrics scrapeable; dashboard renders\n"
        ),
        "labels": ["roadmap", "observability", "prometheus", "grafana"],
    },
    {
        "title": "Agent Layer: RAG + Action Agent (Slack/Jira demo)",
        "body": (
            "Goals:\n"
            "- Agent workflow: query → retrieve → generate → action\n"
            "- Demo tool: Slack message or Jira ticket\n"
            "- README: scenario walkthrough\n\n"
            "Acceptance Criteria:\n"
            "- Deterministic demo using sample corpus\n"
        ),
        "labels": ["roadmap", "agent"],
    },
    {
        "title": "QA: Coverage badge and E2E via httpx",
        "body": (
            "Goals:\n"
            "- pytest-cov + shields.io badge\n"
            "- End-to-end tests with httpx\n\n"
            "Acceptance Criteria:\n"
            "- Coverage >= 80%; E2E green\n"
        ),
        "labels": ["roadmap", "testing"],
    },
    {
        "title": "Docs: Architecture diagram and screenshots",
        "body": (
            "Goals:\n"
            "- Excalidraw/Lucidchart diagram\n"
            "- Screens: MLflow, Langfuse, Grafana\n"
            "- README: Interview questions checklist\n\n"
            "Acceptance Criteria:\n"
            "- Docs render clean; repo professional\n"
        ),
        "labels": ["roadmap", "docs"],
    },
]


def create_issue(
    repo: str, token: str, title: str, body: str, labels: list[str]
) -> None:
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"title": title, "body": body, "labels": labels}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 300:
        print(
            f"Failed to create issue '{title}': {r.status_code} {r.text}",
            file=sys.stderr,
        )
    else:
        print(f"Created: {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed roadmap issues")
    parser.add_argument("--repo", required=True, help="owner/repo")
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN env var is required", file=sys.stderr)
        sys.exit(1)

    for item in ROADMAP:
        create_issue(args.repo, token, item["title"], item["body"], item["labels"])


if __name__ == "__main__":
    main()
