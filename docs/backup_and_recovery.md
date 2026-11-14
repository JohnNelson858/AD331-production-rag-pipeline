# Harbor Cloud backup and recovery

Fictional internal documentation for AD 331.

## Staging backups
Take daily snapshots of staging databases and retain them for seven days. Staging snapshots are intended for test recovery. Use masked or synthetic data in any restored staging database.

## Production backups
Production databases take daily snapshots and retain them for 35 days. Tier 1 services use continuous transaction log backups with a recovery point objective of 15 minutes.

## Tier 1 recovery targets
The recovery time objective for Tier 1 production services is two hours. The recovery point objective is 15 minutes. These are planning targets; record actual recovery time and data loss during exercises.

## Restore verification
Restore into an isolated recovery environment first. Verify row counts, schema version, and a read-only application smoke test before asking the service owner to approve cutover.
