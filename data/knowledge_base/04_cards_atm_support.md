# Horizon Federal Bank: Cards and ATM Support Policy

**Classification:** Synthetic internal demo policy

## Card Security

Agents must never request an OTP, PIN, CVV, password, or complete card number. Identify a card by approved masked information such as the last four digits.

## Common Workflows

### Lost, Stolen, or Compromised Card

1. Verify the customer.
2. Hotlist the card using the approved workflow.
3. Provide `[Hotlist Reference]`.
4. Review recent transactions for possible fraud.
5. Create a replacement request if requested and eligible.

### PIN and Channel Controls

Agents may guide customers to approved self-service reset and card-control channels. Agents must not view, create, communicate, or reset a PIN manually.

### ATM Cash Not Dispensed

**Verified regulatory requirement**

For an ATM transaction where the account is debited but cash is not dispensed, RBI's failed-transaction framework prescribes proactive reversal within T+5 days and Rs 100 per day compensation for delay beyond T+5.

Capture ATM ID/location, date, time, amount, transaction reference, and whether partial cash was dispensed.

### Declined Card Transaction With Debit

Apply the transaction-specific RBI TAT entry. Register the complaint even where automated reversal is in progress.

### Unauthorized Card Transaction

Use the Fraud and Unauthorized Transactions Guide. Do not process it merely as a failed payment.

### Merchant Dispute

Collect evidence and use the current card-network eligibility rules configured in the system. Filing windows vary by network, reason code, and transaction circumstances.

## Suggested Script - Adapt Naturally

> I am sorry the ATM did not dispense the cash. I have registered complaint `[Ticket ID]` for `[Amount]` at ATM `[ATM ID]`. The applicable system-calculated reversal deadline is `[Verified Due Date]`. We will track compensation automatically if the prescribed deadline is breached.

## Sources

- RBI failed-transaction TAT framework, September 20, 2019: https://rbi.org.in/commonman/Upload/English/Notification/PDFs/CIRCULAR677EC931A7A65E4D99AA957D8E85BC0A2A.PDF
- RBI unauthorized transaction liability circular, July 6, 2017: https://rbi.org.in/commonman/Upload/English/Notification/PDFs/NOTI1506072017.PDF

Synthetic demo policy. Compliance validation required before production use.
