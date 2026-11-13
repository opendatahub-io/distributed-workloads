# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in this repository or in
the images it builds, report it to the Red Hat Product Security team:
https://access.redhat.com/security/team/contact

## Threat model and trust boundaries

The training runtime images built by this repository are execution
environments. Customers run their own code in them, and choose which data,
models, and artifacts that code operates on.

We do not consider the following to be vulnerabilities in these images:

- Loading model checkpoints, serialized models, or configuration files created
  by third parties. Loading an untrusted artifact is a trust decision made by
  the customer, in the same way that installing a third-party Python package
  with `pip` is. We do not, and cannot, guarantee that artifacts a customer
  chooses to load are safe.

We do address vulnerabilities in the software we ship and the dependencies we
pre-install in these images. For details of supported images and their
contents, see the
[supported configuration documentation](https://access.redhat.com/articles/rhoai-supported-configs-3.x).
