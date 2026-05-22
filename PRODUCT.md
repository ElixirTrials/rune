# PRODUCT.md

> **Read me first.** This file grounds every non-trivial technical decision Claude makes
> in this repo. Without it, defaults skew toward over-engineering. **Replace every
> `<!-- TODO -->` with real content before relying on Claude for product decisions.**
> The SessionStart hook will warn until all stubs are resolved.
>
> Audience: a head-of-product agent should be able to read this and have an accurate
> mental model of what we're building, for whom, and what we will and will not do.
> Keep concrete and current — out-of-date PRODUCT.md is worse than missing.

## Instructions for Claude when stubs remain

When this file still contains `<!-- TODO -->` markers and the user asks for help filling it in (or starts feature work that needs the missing context), enter **interview mode**:

- Act as a senior head-of-product / product strategist, not a note-taker.
- Interview the user with penetrating, specific questions about their goal, users, and strategy. **One question at a time** — wait for the answer before the next.
- Question ruthlessly. Make them justify decisions. "Why that user, not this one?" "What evidence?" "What happens if you're wrong about that?" "What would change your mind?"
- Surface contradictions between sections (e.g. north-star metric doesn't match the stated jobs-to-be-done; out-of-scope contradicts the persona).
- Push back on vague answers ("everyone", "users", "make it better"). Demand specifics.
- For the *Regulatory surface* section: do not let the user skip framework scoping ("we'll figure out HIPAA later" → no, scope it now or explicitly defer with a written reason).
- For the *Do-not-break invariants* section: stress-test each one. "How would you know it's broken? How loud does it fail?"
- **Do not stop the interview until you have 95% confidence you understand what the user wants to build, for whom, and why.** State your confidence explicitly when you think you're done. If under 95%, name what's still uncertain and keep asking.
- Only after the interview do you draft the PRODUCT.md content. Show the user each section for confirmation before writing. Never invent answers to fill stubs.
- When the user asks "let's just start coding" before stubs are resolved: refuse politely. Cite the SessionStart-hook nag. Offer to compress the interview to the 3-5 most decision-shaping questions if time is short.

---

## 1. North-star metric

One number that, if it goes up, means we are winning. Should be measurable today
or in the near term. One sentence.

<!-- TODO: e.g. "Time from clinician input to validated trial-eligibility decision, p50, in seconds." -->

## 2. Users & personas

Primary users (who *uses* the product), then secondary stakeholders (who *cares* about
the product but doesn't open it). For each: 1-2 sentences on context, motivation,
and what success looks like for them.

- **Primary**: <!-- TODO -->
- **Secondary**: <!-- TODO -->

## 3. Jobs to be done

What hires our product? List 3-5 jobs in the form
"When [situation], I want to [motivation], so I can [outcome]."

<!-- TODO -->

## 4. Regulatory surface

Which frameworks apply to *this* repo. Be specific about scope.

- **HIPAA**: <!-- TODO: in scope? for which services? what PHI is touched? -->
- **GDPR**: <!-- TODO: in scope? lawful basis? data residency? -->
- **FDA / CE / MDR**: <!-- TODO: device class? SaMD? clinical decision support carve-out? -->
- **SOC 2 / ISO 27001**: <!-- TODO: in audit window? -->
- **Vanta program**: <!-- TODO: which frameworks tracked in Vanta? link to workspace. -->

## 5. Do-not-break invariants

Things that, if regressed, would be catastrophic (patient safety, data integrity,
regulatory exposure, trust). Each item: what it is + why it matters + how it fails loudly.

<!-- TODO: e.g. "Audit log entries for any PHI read are append-only and immutable. Loss = HIPAA breach." -->

## 6. Out-of-scope (explicit non-goals)

What we will NOT build, and the reason. As important as the goals — prevents scope creep
and tells Claude "don't 'helpfully' add this."

<!-- TODO: e.g. "Patient self-service portal — out of scope; we are a clinician-facing tool." -->

## 7. Success metrics & current bets

- **Leading metrics** (what we measure weekly): <!-- TODO -->
- **Current bet** (the one thing we're trying to prove this quarter): <!-- TODO -->
- **Kill criteria** (what would cause us to stop the bet): <!-- TODO -->

## 8. Open product questions

Real unresolved questions. These bound where Claude should *ask* rather than *assume*.

<!-- TODO -->

## 9. Glossary

Domain terms with precise definitions. One line each. Saves Claude from inferring.

<!-- TODO: e.g. "Eligibility decision — output of agent-b-service indicating whether a candidate matches a trial protocol." -->
