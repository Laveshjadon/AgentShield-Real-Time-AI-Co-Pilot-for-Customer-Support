# Horizon Federal Bank: Customer Verification and Privacy Policy

**Classification:** Synthetic internal demo policy

## Core Rule

Verify enough to perform the requested action, but collect no more information than necessary.

## Information Matrix

| Information | Agent Treatment |
| --- | --- |
| OTP, PIN, CVV, password | Never request, view, store, or repeat |
| Complete card number | Never request in ordinary support; use masked display |
| Account number | Mask in speech and notes unless the approved workflow requires otherwise |
| PAN | Use only through approved KYC channels; distinguish tax PAN from card Primary Account Number |
| Aadhaar | Never request the full number in open chat or calls; use approved secure workflows |
| Date of birth | Use only as an approved verification factor |
| Registered mobile/email | Confirm using masked values |
| Transaction details | Collect only what is required to identify the transaction |

## Verification Rules

- Use the verification level required for the requested action.
- Do not reveal account existence or details before verification.
- A caller's possession of a phone number is not sufficient verification for high-risk actions.
- Contact-detail changes require stronger controls because they can enable account takeover.
- Third parties, nominees, and representatives receive information only through the applicable authority and verification process.

## Secure Document Handling

Direct customers to an approved secure upload, branch, or authenticated banking channel. Do not request sensitive documents through personal email, messaging applications, or unofficial links.

## Suspected Social Engineering

Stop the requested action, protect the account where authorized, record indicators, and escalate to fraud operations.

## Suggested Script - Adapt Naturally

> For your security, I will never ask for your OTP, PIN, CVV, or password. I can verify you using the approved details already registered with the bank. Please upload any required document only through the secure link shown in your authenticated banking channel.

## Sources

- RBI, *Master Direction - Know Your Customer (KYC) Direction, 2016*, as updated.
- Applicable Indian privacy and information-security requirements must be reviewed by qualified counsel.

Synthetic demo policy. Compliance validation required before production use.
