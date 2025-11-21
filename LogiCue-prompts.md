# LogiCue Prompts

This document contains the pattern-specific prompts used in the LogiCue framework. Each prompt addresses the specific reasoning failure mechanism for its corresponding inference pattern.

---

## CT (Contraposition)

In classical logic, (P → Q) is equivalent to (¬Q → ¬P). This is valid for the material conditional. However, natural language "if" is not a material conditional. It implies a connection or dependence between P and Q.

Therefore, the inference from "If P, then Q" to "If not Q, then not P" is INVALID for natural language.

Counterexample: "If it's raining, then it's not raining hard" can be true (light rain). But its contrapositive, "If it's raining hard, then it's not raining," is a contradiction and is always false.

---

## ASd (Antecedent Strengthening with Defeasibility)

When evaluating conditional statements, remember that real-world conditionals often have implicit background assumptions and normal conditions built in. Don't apply formal logical rules mechanically without considering whether additional conditions in the antecedent might contradict these implicit assumptions.

---

## CMP (Complex Modus Ponens)

Critical reasoning about nested conditionals with probabilities:

Given three options X, Y, Z where P(X) > P(Y) > P(Z), if X doesn't occur, then Y becomes the most likely, NOT Z.

The nested conditional "if X doesn't happen, then if Y doesn't happen, Z will happen" does NOT make "if X doesn't happen, Z will happen" likely. Why? Because when X is eliminated, Y still has higher probability than Z.

Example: If P(X)=0.50, P(Y)=0.35, P(Z)=0.15, then P(Z|not X) = 0.15/(0.35+0.15) = 0.30, which is not likely.

The inference only holds when Z's probability is close to Y's, which contradicts the premise that Z has "much lower odds."

---

## DSmi (Disjunctive Syllogism with Modal Indefinite)

When evaluating logical inferences involving modal expressions like "might," "could," "possibly," etc., remember:

"Might not P" means "possibly not P" - it allows for uncertainty

"Might not P" does NOT mean "not P"

You cannot use disjunctive syllogism (Either A or B + Not B → A) with mere possibilities.

Example: From "Either A or B" + "A might not be true" you CANNOT conclude "B is true" because "might not" only indicates possibility, not definitive negation.

Only definitive statements like "not P" or "definitely not P" can be used in disjunctive syllogism.

---

## DSmu (Disjunctive Syllogism with Modal Unnegation)

When evaluating arguments with "must," distinguish between:
- Content: "Fido is in the garden"
- Modal claim: "Fido must be in the garden" (necessity)

"Either A or B must be true" creates a disjunction between:
1. A
2. The necessity of B (not just B itself)

"It's not the case that B must be true" only negates the necessity - B could still be true, just not necessarily so.

Therefore: "Either A or B must be true" + "B need not be true" does NOT allow you to conclude A.

Example: "Either it's raining or it must be sunny" + "It need not be sunny" ≠ "It's raining"

---

## MTmi (Modus Tollens with Modal Indefinite)

When working with logical inferences, carefully distinguish between definitive statements and statements about possibility/uncertainty. For modus tollens (If P then Q, not Q, therefore not P) to be valid, you need a definitive negation of Q, not just uncertainty about Q.

If given 'If P then Q' and 'Q might not be true' (or 'possibly not Q'), you can only conclude 'P might not be true' - NOT 'P is definitely false.'

Before concluding any logical inference, check: Do I have definitive premises or only statements about possibility? Match your conclusion's certainty level to your premises' certainty level.

---

## MTmu (Modus Tollens with Modal Unnegation)

When evaluating modus tollens with modal terms (must/might):

Standard MT: 'If P then Q' + 'not Q' → 'not P' ✓ VALID

Modal MT: 'If P then Q must be true' + 'Q need not be true' → 'P is false' ✗ INVALID

Key distinction: 'It's not the case that Q must be true' means Q's truth is uncertain/optional, NOT that Q is definitely false. Only definitive negation of Q allows concluding not-P.

Before applying modus tollens, ask: Is the second premise a definitive negation of the consequent, or just a statement about necessity/possibility?

---

## Usage Notes

These prompts are used as system prompts in the LogiCue framework. They provide models with the specific reasoning guidance needed to avoid systematic errors on each pattern.

For implementation details, see `runner.py` where these prompts are defined in the `LOGICUE_PROMPTS` dictionary.
