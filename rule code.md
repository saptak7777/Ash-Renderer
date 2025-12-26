You are extending an EXISTING codebase, not writing from scratch.
Your goal is to produce code that looks exactly like it was written
by a careful, experienced human engineer on this project.

====================================================
1) EXISTING CODE CONTEXT (COPY REAL EXAMPLES HERE)
====================================================

[PASTE 2–5 SHORT EXTRACTS FROM CURRENT CODEBASE]

For example:
- 1–2 typical functions
- 1 test module
- 1 example of error handling
- 1 example of documentation + comments

From these examples, infer and FOLLOW exactly:

A. Error handling pattern
   - How errors are represented (Result<T, E>? Custom error type?)
   - Where panics are allowed vs avoided
   - How input is validated
   - Whether unwrap/expect is used, and in what situations

B. Naming conventions
   - Function names (verb_noun? snake_case?)
   - Type/struct/enum names (CamelCase?)
   - Variable names (short vs descriptive, domain terms used?)
   - Constant names (SCREAMING_SNAKE_CASE?)

C. Code structure & style
   - Typical function length (approx. max lines)
   - Typical nesting depth (if/loop/match depth)
   - Use of early returns vs single return at end
   - Preference for iterators vs for-loops
   - Module and file organization patterns

D. Documentation & comments
   - Presence and style of doc comments (/// …)
   - Whether comments explain WHY vs WHAT
   - How examples are written (doctests? pseudo-code?)
   - Tone: formal, concise, casual?

E. Testing patterns
   - How tests are named
   - Typical structure: happy-path / edge-case / error-case?
   - Use of property-based tests (e.g., proptest) or just unit tests
   - How fixtures or test data are defined

F. Defensive programming & security
   - How untrusted input is handled
   - Typical validation style and depth
   - Where assumptions are documented (// SAFETY, // PRE, etc.)
   - How error messages are phrased

G. Performance philosophy
   - Is clarity preferred over micro-optimizations?
   - Are allocations/clones avoided or accepted for readability?
   - How hot paths are treated (comments about performance?)

You MUST match all these patterns as closely as possible.

====================================================
2) NEW FEATURE / CHANGE REQUEST
====================================================

Describe the change precisely:

FEATURE / CHANGE:
- Goal: [What should this new feature do?]
- Scope: [Which module/file/function(s) does it touch?]
- Inputs: [Types, trusted/untrusted, ranges, formats]
- Outputs: [Return types, side effects, invariants]
- Errors: [What can go wrong? How should each case behave?]
- Edge cases: [Empty input, boundary values, odd but valid inputs]
- Security concerns: [Validation needed? Permissions? Sensitive data?]
- Performance constraints: [Latency/memory/throughput if important]

Example:
- Add a rate limiter to the API request handler.
- Inputs: user_id, endpoint, timestamp (all untrusted).
- Output: Ok(()) if under limit, Err(RateLimitError) if over.
- Edge cases: first request, clock skew, bursty traffic.
- Must integrate cleanly with existing error type: AppError.

====================================================
3) PARAMETERS FOR “HUMAN-QUALITY” CODE
====================================================

When generating or modifying code, you MUST:

1. **Match error handling exactly**
   - Use the SAME error type(s) as in the examples (e.g., AppError, MyError).
   - Use Result<T, E> exactly the same way.
   - Only panic where existing code would panic (invariants, impossible states).
   - Avoid unwrap/expect on untrusted data unless examples clearly do this,
     and then include a clear SAFETY/assumption comment.

2. **Match naming conventions**
   - Follow the exact function naming pattern (e.g., verb_noun: parse_config).
   - Use domain-specific names, not generic “data”, “tmp”, “foo”.
   - Keep local variable naming style consistent (short vs descriptive).
   - Use the same casing and style for types, modules, and constants.

3. **Match code structure**
   - Keep functions roughly the same size and complexity as in the examples.
   - Respect typical nesting depth (e.g., no deeply nested 5-level ifs
     if existing code prefers early returns).
   - Stick to similar patterns: if existing code uses iterators + map/filter,
     do the same; if it uses classic for loops, mirror that.
   - Do NOT reorganize existing logic unless explicitly asked.

4. **Match documentation style**
   - If examples use doc comments (///) with sections (Arguments, Returns, Errors),
     do the same.
   - If comments are sparse and only explain “why” something is done,
     follow that: don’t over-document obvious code.
   - Use similar tone and level of detail.
   - If examples include doctests, add similar examples for new public APIs.

5. **Match testing style**
   - Add tests in the same file/module or test layout as existing tests.
   - Use same naming pattern for tests (e.g., test_feature_xxx).
   - For each major function, add at minimum:
     - One happy-path test
     - One or more edge-case tests
     - One or more error-case tests
   - If property-based tests are used in existing code, mirror that style
     for the new behavior.

6. **Match defensive programming & security posture**
   - Validate inputs at the same layer(s) and in the same way (e.g., length/range checks).
   - Use the same pattern for authentication/authorization as existing code.
   - Avoid introducing new trust assumptions different from existing ones.
   - For any assumptions, add comments in the same style used in examples
     (// SAFETY, // PRECONDITION, etc.).

7. **Match performance philosophy**
   - If the project clearly prefers clarity over small optimizations,
     don’t micro-optimize at the cost of readability.
   - If hot paths are optimized in existing code, treat similar paths likewise
     and document trade-offs similarly.
   - Don’t introduce heavyweight abstractions where existing code is simple.

8. **Locality and minimal change**
   - Touch only the files/functions necessary for this feature.
   - Do NOT refactor unrelated code unless explicitly requested.
   - Keep diffs minimal and focused on the new behavior.

9. **Style & formatting**
   - Assume code will be run through the project’s formatter, but your structure
     should already look idiomatic and consistent.
   - No huge “god functions” if the codebase splits logic into helpers.
   - No introducing new architectural patterns (e.g., adding a new layer)
     unless asked.

====================================================
4) OUTPUT EXPECTATIONS
====================================================

Your response should include:

1. **Updated or new functions**
   - Full code blocks, ready to paste.
   - Match existing style in signatures, visibility (pub / private), and layout.

2. **Tests**
   - Unit tests covering happy-path, edge cases, and error cases
     in the same style as existing tests.
   - Place tests where existing tests live (e.g., #[cfg(test)] mod tests { … }).

3. **Documentation & comments**
   - Doc comments for any new public APIs, matching existing style.
   - Inline comments only where they explain non-obvious decisions or assumptions.

4. **Short rationale**
   - Briefly explain how the new code:
     - Matches the existing patterns (error handling, naming, structure, tests)
     - Handles edge cases and errors
     - Maintains performance and security expectations
   - Keep this explanation separate from the code blocks.

====================================================
5) IMPORTANT GUARDRAILS
====================================================

DO NOT:
- Change existing function signatures or public APIs unless explicitly requested.
- Introduce new error types or logging systems unless required by the spec.
- Reformat or rewrite unrelated parts of the code.
- “Improve” patterns just because you think they’re better; match what’s there.

MUST:
- Keep behavior backwards compatible unless told otherwise.
- Preserve the project’s idioms, even if they’re slightly imperfect.
- Produce code that a senior engineer could plausibly have written
  in the same codebase, at the same time.

Now, based on all of the above, implement the requested feature/change.
Return ONLY code and minimal explanation, no extra commentary.
