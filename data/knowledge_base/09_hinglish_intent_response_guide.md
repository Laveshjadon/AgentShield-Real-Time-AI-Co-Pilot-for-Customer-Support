# Horizon Federal Bank: Hinglish Intent and Response Guide

**Classification:** Synthetic training and retrieval data

## Usage

These examples help identify Roman-script Hinglish intent. Responses are adaptable examples, not regulatory policy.

| Customer utterance | English meaning | Intent | Risk | Suggested response |
| --- | --- | --- | --- | --- |
| Mere account se paise kat gaye lekin UPI payment fail dikha raha hai. | Money was debited but UPI failed. | Failed UPI | Medium | Main transaction status check karke complaint register karta hoon. Aapka reference `[Ticket ID]` hoga aur verified reversal date `[Due Date]` hai. |
| Yeh transaction maine nahi kiya. Card turant block karo. | I did not make this transaction; block my card. | Unauthorized card transaction | High | Main card ending `[Last 4]` ko secure kar raha hoon. OTP, PIN ya CVV share mat kijiye. Hotlist reference `[Reference]` hai. |
| Mera phone hack ho gaya aur banking app open nahi ho raha. | Phone may be compromised. | Account takeover | Critical | Main protective restriction workflow start kar raha hoon aur fraud team ko escalate karunga. Abhi kisi ko OTP ya password mat batayein. |
| EMI miss ho gayi, kya CIBIL kharab ho jayega? | EMI was missed; asks about credit score. | EMI support | Medium | Main aapke loan ka verified payment status aur agreement check karta hoon. Main exact score impact predict nahi karunga, lekin available payment aur support options explain karunga. |
| Merchant refund bol raha hai par paisa nahi aaya. | Merchant says refund was sent but it is missing. | Merchant refund | Low | Refund reference aur expected date check karke trace request bana dete hain. Aapka ticket `[Ticket ID]` rahega. |
| Tumhari service bahut kharab hai, koi help nahi karta. | Service is poor; nobody helps. | Complaint/frustration | Warning | Mujhe afsos hai ki aapko baar-baar contact karna pada. Main existing complaint history check karke same case ko properly escalate karta hoon. |
| Main RBI mein complaint karunga. | Customer will complain to RBI. | Regulatory escalation | Warning | Aapko complaint escalate karne ka adhikar hai. Main pehle bank complaint reference aur current status clearly provide karta hoon. |
| Main tum sabko dekh lunga, office aa raha hoon. | Possible threat to staff. | Threat | Critical | Main aapki concern register kar raha hoon, lekin threatening language continue hone par safety escalation follow karna hoga. |

## Classification Rules

- Complaint about service: not automatically abusive.
- Threat to complain to RBI or court: regulatory escalation, not physical threat.
- “Fraud company” may indicate hostility but requires context.
- Threats of violence or harm: critical safety escalation.
- Self-harm language: follow the dedicated safety protocol; do not treat it as ordinary toxicity.

Synthetic demo policy. Compliance validation required before production use.
