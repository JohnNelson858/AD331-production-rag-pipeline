# Harbor Cloud billing policy

Fictional internal documentation for AD 331.

## Staging budget alerts
Each staging project has a monthly budget of USD 300 by default. Notify the project owner at 75 percent of budget and notify both the owner and FinOps at 90 percent. Budget alerts do not stop resources automatically.

## Resource allocation tags
Every billable resource must have cost_center, service_owner, and environment tags. FinOps sends unallocated charges to the account owner for review and tracks corrections in the monthly report.

## Idle compute
A staging compute instance with no workload for seven consecutive days is flagged for review. The owner receives two business days to respond before the platform team schedules shutdown.

## Budget exceptions
Request a temporary budget increase through FinOps. Include the cost estimate, reason, and expiration date. Approval changes the budget threshold for the stated period.
