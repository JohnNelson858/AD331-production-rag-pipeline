# Harbor Cloud architecture

Fictional internal documentation for AD 331.

## Staging isolation
Staging runs in a separate cloud account from production. To provision an isolated staging VPC, use the harbor-vpc Terraform module version 3.4 or later and set environment to staging and network_mode to isolated. Use a CIDR from the staging address registry.

## Private subnet routing
An isolated staging VPC has no direct internet route. Application subnets use private service endpoints for the container registry, object storage, and logs. Adding public egress requires a separate network review.

## Legacy Terraform modules
Harbor VPC module 2.x does not support network_mode=isolated. Teams migrating from 2.x must create a new VPC with module 3.4 or later, validate private endpoints, and move workloads during a scheduled staging window.

## Production network changes
Production uses a separate Terraform state and approval workflow. A staging approval does not cover production. The Network Operations team reviews connectivity and the service owner approves the rollout.
