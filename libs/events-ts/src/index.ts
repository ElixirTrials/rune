/**
 * Event types and helpers for TypeScript services (e.g. Pub/Sub).
 */

export type EventKind = "created" | "updated" | "deleted";

/** Base event envelope. */
export interface EventEnvelope<T = unknown> {
  id: string;
  kind: EventKind;
  payload: T;
  timestamp: string;
}

/**
 * Create an event envelope.
 *
 * @param kind - Event kind
 * @param payload - Event payload
 * @returns Event envelope
 */
export function createEvent<T>(
  kind: EventKind,
  payload: T,
  id?: string
): EventEnvelope<T> {
  return {
    id: id ?? crypto.randomUUID(),
    kind,
    payload,
    timestamp: new Date().toISOString(),
  };
}
