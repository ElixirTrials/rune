/**
 * Shared utility functions.
 */

/**
 * Check if a value is defined (not null and not undefined).
 *
 * @param value - Value to check
 * @returns True if defined
 */
export function isDefined<T>(value: T | null | undefined): value is T {
  return value != null;
}
