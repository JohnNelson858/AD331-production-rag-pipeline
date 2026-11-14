# Harbor Cloud deployment runbook

Fictional internal documentation for AD 331.

## Staging rollout
Deploy an immutable container image by digest to staging. Run the health check against /healthz and the application smoke tests. Record the image digest and test results in the change ticket before requesting production approval.

## Production canary
Start the production canary at 5 percent of traffic. Observe it for 15 minutes before widening the rollout. The on-call engineer checks the HTTP 5xx rate and p95 latency against the baseline recorded in the change ticket.

## Rollback criteria
Roll back if the canary HTTP 5xx rate exceeds 2 percent for five consecutive minutes or if p95 latency exceeds the baseline by 30 percent for ten minutes. Restore the previous image digest and repeat the smoke test.

## Database changes
Apply backward-compatible schema changes before deploying application code. Remove deprecated columns only in a later change after verifying that no active version depends on them.
