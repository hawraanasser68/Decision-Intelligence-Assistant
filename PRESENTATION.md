# Decision Intelligence Assistant — Presentation

**Total time: ~10 minutes**

---

## SLIDE 1 — Title

**Decision Intelligence Assistant**
*What if AI could help support agents make better decisions, faster?*

AIE Assignment 3 | April 2026

---
**SCRIPT:**
> "What if every time a support agent opened a new customer complaint, they had a smart assistant sitting next to them — one that had read every single ticket the team had ever handled, and could instantly say: this looks urgent, here's how we've handled this before, and here's a draft reply. That's what this project is about."

---

## SLIDE 2 — Table of Contents

1. The problem — a day in the life of a support agent
2. The idea — what if AI could help?
3. Who benefits
4. How we built it — simply explained
5. What surprised us along the way
6. See it in action

---
**SCRIPT:**
> "Here's where we're going. We'll start with a real problem, build up to the solution, talk about who it helps, quickly explain what's under the hood, share what we learned, and then I'll show you the live app."

---

## SLIDE 3 — The Problem

**Imagine you're a customer support agent.**

It's Monday morning. 300 new tickets just came in overnight.

- One customer has been without internet for 3 days — furious.
- Another has a billing question.
- Another just wants to say thank you.

You have to read every single one, decide which is most urgent, and write a helpful reply — all before the next wave arrives.

**The reality:**
- Urgent tickets get missed
- Replies become copy-paste and generic
- New agents don't know how previous cases were handled
- The team is overwhelmed

---
**SCRIPT:**
> "Picture this. You're a support agent. It's Monday morning and 300 tickets came in while you were asleep. You have no idea which ones are fires. You don't have time to read them all carefully. So you skim, you prioritise by gut feeling, and you write the same generic reply you've written a hundred times. Somewhere in that pile, a customer has been without service for three days. They're about to post on social media. You missed them. This is the problem — and it's happening at every support team, every day."

---

## SLIDE 4 — The Idea

**What if AI could be that experienced colleague looking over your shoulder?**

One who has read every past ticket, knows which ones blew up, and can instantly say:

- *"This one's urgent — 87% confident"*
- *"Here's how we handled similar complaints before"*
- *"Here's a draft reply based on what actually worked"*

That's the Decision Intelligence Assistant.

> Not replacing the agent. Giving them a head start.

---
**SCRIPT:**
> "The idea is simple: give every agent an AI co-pilot. Not one that takes over — one that gives them a head start. The agent still makes the final call. But instead of starting from zero, they get an instant urgency signal and a draft reply that's grounded in real past cases. The AI handles the pattern-matching. The human handles the judgment."

---

## SLIDE 5 — Who Benefits?

**Think of it as a ripple effect.**

- **The agent** spends less time deciding what to write — more time on complex cases
- **The team lead** stops worrying about urgent tickets slipping through
- **The customer** gets a faster, more relevant answer — not a template
- **The business** sees lower handling time, fewer escalations, happier customers

> When you help the agent, everyone downstream wins.

---
**SCRIPT:**
> "The benefits aren't just for the agent. When agents are faster and more consistent, customers wait less and feel heard more. When urgent tickets are flagged automatically, team leads can stop playing whack-a-mole. And when the business sees fewer escalations and better satisfaction scores, that's a real, measurable return. It all starts with giving the agent a better tool."

---

## SLIDE 6 — How It Works

**You type a customer message. Four things happen at once.**

Think of it as four colleagues answering at the same time:

1. **The statistician** — a trained ML model scans the message for urgency signals: how negative is the tone? How long has this been going on? Keywords like "urgent", "days", "broken". Gives a priority label with a confidence score.

2. **The generalist** — a large language model (GPT-4o-mini) reads the message cold, with no background. Gives a priority prediction and drafts a reply from general knowledge alone.

3. **The archivist** — searches 284,000 real past support interactions for the 5 most similar cases. Pulls up what agents actually said when they faced the same thing.

4. **The grounded writer** — the same LLM, but now it reads those 5 real cases first, then writes a reply based on what actually worked before.

**The result:** four outputs, side by side, in under 3 seconds.

---
**SCRIPT:**
> "Here's how to picture it. You type a message and four things happen simultaneously. A trained machine learning model scores the urgency — it's looked at the tone, the keywords, the structure of the message. Separately, a large language model reads the message with no context and gives its own answer and priority guess. Meanwhile, a search engine is scanning 284,000 real customer support conversations for the closest matches. And then that same LLM reads those real cases and writes a reply that's grounded in actual resolutions — not invented from thin air. Four perspectives. One screen. Three seconds."

---

## SLIDE 7 — What Surprised Us

**We didn't just build the system — we learned from it.**

**Surprise 1: AI makes things up — unless you stop it**
When we first ran the system, the LLM confidently told a customer about a "30-day return policy." That policy doesn't exist anywhere in our data. It invented it. Once we told the model to only use retrieved cases and nothing else, the hallucination stopped. The fix was in the instructions — not the model.

**Surprise 2: When AI disagrees with itself, pay attention**
The ML model and the LLM usually agree on urgency. But sometimes they don't. And when they don't? It's almost always on a genuinely tricky case — one where a human really should take a second look. The disagreement is a feature, not a bug.

**Surprise 3: Confidence isn't always what it seems**
The ML model says "55% urgent" and means it — it genuinely can't decide. The LLM says "90% urgent" on the same case. Who's right? Neither, necessarily. Understanding where confidence comes from matters.

**Surprise 4: Your data shapes your AI's personality**
Our support data came from Twitter — short, informal replies. So our RAG answers came back sounding like tweets: "Hey! DM us and we'll sort this." That's not bad — it's just honest. A richer dataset would produce richer answers.

---
**SCRIPT:**
> "Building this taught us things we didn't expect. The biggest one: AI will confidently make things up unless you explicitly stop it. We caught GPT inventing a return policy that wasn't in our data at all. A small change to the prompt — 'only use what's in the cases, nothing else' — fixed it completely. That's a lesson that generalises well beyond this project. The second thing: when our two AI approaches disagree on priority, it's almost always on a case that genuinely is ambiguous. That disagreement is actually valuable — it tells the agent 'this one needs a human.' And third: the quality of what you get out is shaped by the quality of what goes in. Twitter-style data gives you Twitter-style answers. That's not a failure — it's just how AI works."

---

## SLIDE 8 — See It in Action

**[LIVE DEMO — 2-3 minutes]**

**Step 1 — Open the app at** `http://localhost:5173`

---

**Demo Query 1:**
> *"My internet has been down for 3 days and nobody is helping me. This is completely unacceptable!"*

**What to point out:**
- Both the ML model and the LLM flag this as **urgent** — the anger is unmistakable
- The LLM-only answer is polished but generic — "we're sorry to hear this"
- The RAG answer reflects how real agents actually responded to similar complaints
- Look at the 5 retrieved cases — the AI is showing its work

---

**Demo Query 2:**
> *"I ordered the AirPods Pro last Tuesday and the left earbud has no sound. Order #APL-2024-88421. What's your return policy?"*

**What to point out:**
- The LLM without context may mention a specific return policy — which it invented
- The RAG answer is honest: "please DM us with your order details" — because that's exactly what the real cases show
- Same LLM, different instructions, very different result
- This is the core argument for RAG: it keeps AI grounded

---
**SCRIPT:**
> "Let me show you the app. [open browser] I'll type in a frustrated customer who's been without internet for three days. [submit] — straight away, both our AI systems flag this as urgent. Notice the ML model gives a confidence score based on statistical probabilities. The LLM gives its own guess. Now look at the answers. The LLM gives a perfectly decent generic reply. The RAG answer is shorter, more direct — because that's how real agents actually talked to customers in similar situations. Now let me try a trickier one. [type second query, submit] — watch the LLM-only answer. It might mention a specific return policy. That policy doesn't exist in our data. The RAG answer doesn't do that — it tells the customer to DM their details, because that's what the real cases show. Same AI. The difference is whether it's grounded in real evidence or not. That's the whole point."

---

## SLIDE 9 — The Takeaway

**Three approaches. Each one incomplete alone. Together, they're powerful.**

- The **ML model** is fast, explainable, and grounded in data — but it can't write a reply
- The **LLM alone** is fluent and flexible — but it invents things it doesn't know
- **RAG** combines both: LLM fluency, real-world grounding, with evidence you can trace

**The bigger idea:**
AI doesn't replace the agent's judgment. It gives them better information to judge with.

> *"Good decisions come from good information. This system makes sure the agent always has it."*

---
**SCRIPT:**
> "So here's what I want you to take away. No single AI approach is enough on its own. The ML model is fast and honest about uncertainty but can't hold a conversation. The LLM is articulate but will confidently fill in gaps with things it made up. RAG is the bridge — it gives the LLM a memory, a set of real past cases to draw on. But even then, the agent is still in the loop, still making the final call. The system doesn't decide. It informs. And better information leads to better decisions. That's what decision intelligence means."

---

*End of presentation — approx. 10 minutes*
