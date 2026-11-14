# Harbor Cloud incident response

Fictional internal documentation for AD 331.

## Severity levels
Declare SEV-1 for widespread production unavailability or confirmed data loss. Declare SEV-2 for significant degradation with a viable workaround. Staging outages ordinarily use SEV-3.

## Incident roles
The incident commander coordinates the response and assigns an operations lead. The communications lead posts status updates. An engineer assigned to repair work should hand off communications before making changes.

## Status update cadence
During an active SEV-1, publish an internal update every 15 minutes. State the user impact, what the team has verified, and when to expect the next update. Label suspected causes as unconfirmed.

## Incident closure
Close the incident after service metrics have returned to baseline and the owner has verified recovery. Schedule a postmortem within three business days and assign owners to follow-up actions.
