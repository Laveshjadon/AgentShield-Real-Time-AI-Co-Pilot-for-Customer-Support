# Horizon Federal Bank: Fraud and Unauthorized Transactions Guide

**Classification:** Synthetic internal demo policy  
**System:** AgentShield is the internal agent-assistance CRM, not the bank  
**Review requirement:** Compliance validation is required before production use.

## Purpose

This guide helps frontline agents contain suspected fraud, register complaints, explain customer-liability rules accurately, and escalate cases without promising a refund or investigation outcome.

## Immediate Protective Actions

### Lost or Compromised Card

**HFB demo policy**

1. Verify the customer using approved non-sensitive factors.
2. Never request an OTP, PIN, CVV, password, or complete card number.
3. Hotlist the affected card and record the last four digits.
4. Register a fraud complaint and provide `[Hotlist Reference]` and `[Ticket ID]`.
5. Review linked channels for additional compromise indicators.

**Suggested script - adapt naturally**

> I understand this is urgent. I am securing the card ending in `[Last 4 Digits]` now. Please do not share any OTP, PIN, CVV, or password with me or anyone else. Your hotlist reference is `[Hotlist Reference]`, and your complaint reference is `[Ticket ID]`.

### UPI Fraud or Account Takeover

**HFB demo policy**

- Block or restrict the affected UPI handle when permitted by the verified workflow.
- For suspected account takeover, place the approved outward-transaction restriction and escalate to the Fraud Investigation Unit.
- Record transaction IDs, timestamps, amounts, beneficiary details, devices, and the customer's account of events.
- Advise customers to use India's official cybercrime reporting channels where appropriate. Do not imply that external reporting guarantees recovery.

## Customer Liability

**Verified regulatory requirement**

RBI's July 6, 2017 circular distinguishes:

- Bank contributory fraud, negligence, or deficiency: zero customer liability.
- Third-party breach reported within three working days of receiving the bank's communication: zero customer liability.
- Third-party breach reported within four to seven working days: limited liability under the applicable RBI table.
- Reports after seven working days: liability follows the bank's Board-approved policy.
- Customer negligence, such as sharing payment credentials: the customer bears loss until reporting; loss after reporting is borne by the bank.
- The bank must provide a shadow reversal within ten working days after notification, subject to the circular's conditions.

Agents must not reduce this framework to “reported in three days means automatic refund.”

## Escalation Triggers

- Suspected account takeover
- Multiple affected payment channels
- Continuing unauthorized activity after blocking
- Vulnerable customer requiring assisted handling
- Customer disputes the assigned liability category
- Insider involvement or systemic compromise suspected
- Frontline agent cannot complete an immediate protective action

## Prohibited Statements

- “You will definitely receive a refund.”
- “This happened because you were careless.”
- “Share the OTP so I can reverse it.”
- “The investigation will finish by `[unverified date]`.”

## Sources

- RBI, *Customer Protection - Limiting Liability of Customers in Unauthorised Electronic Banking Transactions*, DBR.No.Leg.BC.78/09.07.005/2017-18, July 6, 2017: https://rbi.org.in/commonman/Upload/English/Notification/PDFs/NOTI1506072017.PDF

Synthetic demo policy. Compliance validation required before production use.
