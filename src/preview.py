"""Repeatable sample traces for inspecting the UI without model downloads or API calls."""

from .schemas import RAGResponse, RetrievedChunk, Turn

EXAMPLES = {
    "internal": "What are the staging requirements for Kubernetes Worker Nodes?",
    "followup": "What are its production requirements?",
    "web": "What is the standard port for DNS?",
}

TOPICS = [
    ("kubernetes_worker_nodes", "Staging worker requirements",
     "Staging workers require 4 vCPUs, 16 GiB RAM, and an 80 GiB encrypted root disk."),
    ("kubernetes_worker_nodes", "Production worker requirements",
     "Production workers require 8 vCPUs, 32 GiB RAM, and a 120 GiB encrypted root disk."),
    ("cloud_architecture", "Staging isolation", "Use harbor-vpc 3.4 or later for an isolated staging VPC."),
    ("kubernetes_worker_nodes", "Worker image upgrades", "Test the replacement image on a staging pool first."),
    ("cloud_architecture", "Private subnet routing", "Isolated staging VPCs have no direct internet route."),
    ("kubernetes_worker_nodes", "Draining worker nodes", "Cordon the worker before draining it."),
    ("cloud_architecture", "Legacy Terraform modules", "Harbor VPC 2.x lacks isolated network mode."),
    ("cloud_architecture", "Production network changes", "Production uses a separate approval workflow."),
    ("deployment_runbook", "Staging rollout", "Deploy the container image by digest."),
    ("deployment_runbook", "Production canary", "Start the production canary at 5 percent traffic."),
    ("deployment_runbook", "Rollback criteria", "Monitor HTTP 5xx rates and p95 latency."),
    ("deployment_runbook", "Database changes", "Apply compatible schema changes first."),
    ("incident_response", "Severity levels", "Widespread unavailability is SEV-1."),
    ("incident_response", "Incident roles", "The incident commander coordinates response."),
    ("incident_response", "Status update cadence", "Update every 15 minutes during SEV-1."),
    ("billing_policy", "Staging budget alerts", "The default staging budget is USD 300."),
    ("billing_policy", "Resource allocation tags", "Record the owner, environment, and cost center."),
    ("backup_and_recovery", "Staging backups", "Keep staging snapshots for seven days."),
    ("backup_and_recovery", "Production backups", "Keep production snapshots for 35 days."),
    ("backup_and_recovery", "Restore verification", "Verify the restored database before cutover."),
]


def preview(question: str, history: list[Turn]) -> RAGResponse:
    case = next((k for k, v in EXAMPLES.items() if v.casefold() == question.casefold()), None)
    if case is None:
        raise ValueError("Choose one of the sample questions, or switch to Live for a custom question.")
    if case == "followup" and not any("Kubernetes Worker Nodes" in t.question for t in history):
        raise ValueError("Run the internal example first and enable conversation before the follow-up.")

    stage1 = [
        RetrievedChunk(chunk_id=f"sample-{i}", title=title, content=body, source="internal_doc",
                       location=f"docs/{file}.md — {title}", dense_rank=i,
                       distance=round(0.2 + 0.021 * i, 3))
        for i, (file, title, body) in enumerate(TOPICS, 1)
    ]
    order = [1, 4, 2, 6, 3] if case == "internal" else [2, 6, 1, 4, 8]
    scores = [0.94, 0.73, 0.62, 0.42, 0.37] if case != "web" else [0.19, 0.14, 0.12, 0.07, 0.04]
    stage2 = [stage1[i - 1].model_copy(update={"score": score, "rerank_rank": rank})
              for rank, (i, score) in enumerate(zip(order, scores), 1)]
    sources = []
    evidence = stage2[0]
    if case == "web":
        evidence = RetrievedChunk(
            chunk_id="sample-web-1", title="IANA Service Name and Port Number Registry",
            content="The registry lists domain (Domain Name Server) on port 53 for TCP and UDP.",
            source="web",
            location="https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=domain",
        )
        sources = [evidence]

    return RAGResponse(
        answer=f"{evidence.content} [1]",
        citations=[evidence.location if case == "web" else evidence.chunk_id],
        route_taken="WEB_FALLBACK" if case == "web" else "INTERNAL_RAG",
        confidence_score=scores[0], threshold=0.55,
        transformed_query="What are the production requirements for Kubernetes Worker Nodes?" if case == "followup" else question,
        rewrite_status="rewritten" if case == "followup" else "unchanged",
        stage1=stage1, stage2=stage2, web_sources=sources,
    )
