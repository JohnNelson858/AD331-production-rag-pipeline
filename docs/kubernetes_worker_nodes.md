# Kubernetes Worker Nodes

Fictional internal documentation for AD 331.

## Staging worker requirements
Each general-purpose staging worker node requires at least 4 vCPUs, 16 GiB of RAM, and an 80 GiB encrypted root disk. Worker nodes must use the approved Harbor Linux image and register with the staging cluster.

## Production worker requirements
Production general-purpose workers require at least 8 vCPUs, 32 GiB of RAM, and a 120 GiB encrypted root disk. Production pools span at least two availability zones.

## Worker image upgrades
Test the replacement image on a staging pool first. Verify pod scheduling, image pulls, and log delivery. In production, replace workers one at a time and respect workload disruption budgets.

## Draining worker nodes
Cordon the worker before draining it. If a disruption budget blocks eviction, pause and contact the workload owner. Record the node name and replacement status in the maintenance ticket.
