# Horizon Federal Bank: Refund and Failed Payment Policy

**Classification:** Synthetic internal demo policy  
**Review requirement:** Validate current RBI, NPCI, and payment-network rules before production use.

## Definitions

- **Failed transaction:** The payment did not complete but one side may have been debited.
- **Unauthorized transaction:** The customer denies authorizing the transaction.
- **Merchant refund:** A merchant has agreed to return funds.
- **Chargeback:** A formal card-network dispute, not a standard bank refund.

Agents must select the correct case type before communicating a timeline.

## UPI and IMPS Failures

**Verified regulatory requirement**

Under RBI's failed-transaction TAT framework, where a person-to-person fund transfer debits the sender but does not credit the beneficiary, reversal is due no later than T+1 day. Compensation under the framework is Rs 100 per day of delay beyond the prescribed deadline.

`T` means the transaction date. The agent must use the deadline calculated by the approved system and must not manually reinterpret holiday treatment.

**Agent action**

1. Retrieve the transaction using `[Transaction ID]`, date, and amount.
2. Confirm whether it is failed, pending, successful, or beneficiary-not-credited.
3. Register `[Ticket ID]`.
4. Communicate the system-calculated due date.
5. Escalate when the prescribed deadline has passed.

**Suggested script - adapt naturally**

> I can see the transfer of `[Amount]` dated `[Date]`. I have registered complaint `[Ticket ID]`. The applicable reversal deadline shown by our system is `[Verified Due Date]`. If the credit is not received by that date, the case will move to overdue escalation automatically.

## NEFT Returns

**Verified regulatory requirement**

If the beneficiary cannot be credited, the destination bank is required to return the NEFT transaction within two hours after completion of the relevant batch. Delayed credit or return attracts penal interest at the current RBI LAF Repo Rate plus two percent for the delay period.

Do not apply the Rs 100-per-day failed-payment compensation rule to NEFT unless an official provision specifically requires it.

## Declined Card Transaction With Debit

Use the RBI failed-transaction TAT table applicable to the transaction type. Do not confuse a declined-but-debited card transaction with an unauthorized card dispute.

## Merchant Refunds and Chargebacks

**HFB demo policy**

- First verify whether the merchant initiated a refund.
- Collect the merchant reference, expected refund date, receipt, and correspondence.
- If eligible, route a card dispute through the applicable network process.
- Network filing windows and resolution periods must come from the current card-network rule set configured in the system.
- Never describe a card-network timeline as an RBI refund mandate.

## Provisional Credit

**HFB demo policy**

Frontline agents may submit a provisional-credit review request but cannot approve it. Eligibility, amount, reversibility, and timing are determined by the authorized disputes team.

> I can submit a provisional-credit review under reference `[Reference ID]`. Approval is not guaranteed and depends on the investigation team's assessment.

## Sources

- RBI, *Harmonisation of Turn Around Time and customer compensation for failed transactions using authorised Payment Systems*, DPSS.CO.PD No.629/02.01.014/2019-20, September 20, 2019: https://rbi.org.in/commonman/Upload/English/Notification/PDFs/CIRCULAR677EC931A7A65E4D99AA957D8E85BC0A2A.PDF
- RBI NEFT customer FAQ: https://www.rbi.org.in/

Synthetic demo policy. Compliance validation required before production use.
