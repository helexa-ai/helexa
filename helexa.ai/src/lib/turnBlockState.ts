/**
 * When the assistant's working is shown, and when it gets out of the way.
 *
 * Kept in its own module, free of React and icon imports, so the rule can
 * be imported and exercised on its own. It is the entire behavioural
 * contract of `TurnBlocks`, and as an inline expression inside the
 * component the only way to check it was to click around.
 */

/**
 * Is this the block the assistant is working in right now?
 *
 * Only the last block can be live, and only while the turn is streaming
 * with no answer text yet. Once answer text starts, the working becomes
 * history and nothing is live — including the block that was live a
 * token earlier.
 */
export function isLive(
  index: number,
  count: number,
  streaming: boolean,
  answerStarted: boolean,
): boolean {
  return streaming && !answerStarted && index === count - 1;
}

/**
 * Should this block be rendered open?
 *
 * `override` is the user's own click, and it wins outright. Someone who
 * closes a noisy reasoning stream should not have it spring back open on
 * the next token, and someone who opens a finished block should keep it
 * open while the rest of the turn streams past.
 */
export function isOpen(
  index: number,
  count: number,
  streaming: boolean,
  answerStarted: boolean,
  override: boolean | undefined,
): boolean {
  return override ?? isLive(index, count, streaming, answerStarted);
}
