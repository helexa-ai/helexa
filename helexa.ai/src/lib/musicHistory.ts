// Grouping for the /music history sidebar (#402): songs filed under a
// project are listed there; everything else falls into date sections, like
// a mail client. Pure functions, so the bucketing is testable without a DOM.

import type { AudioJob } from "./musicClient";

export type Period = "today" | "yesterday" | "week" | "month" | "older";

/** Date sections in display order. */
export const PERIODS: Period[] = ["today", "yesterday", "week", "month", "older"];

/**
 * The section a job created at `createdAt` (unix seconds) falls in, by the
 * viewer's local calendar days, not 24-hour windows: something made at
 * 23:50 is "yesterday" ten minutes later.
 */
export function periodOf(createdAt: number, now: Date): Period {
  const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 86_400_000;
  const t = createdAt * 1000;
  if (t >= midnight) return "today";
  // Calendar arithmetic rather than `midnight - day`, so a DST change
  // between the two days cannot shift the boundary by an hour.
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime();
  if (t >= yesterday) return "yesterday";
  if (t >= midnight - 7 * day) return "week";
  if (t >= midnight - 30 * day) return "month";
  return "older";
}

export interface HistoryGroups {
  /** Project id → its songs, newest first. */
  byProject: Map<string, AudioJob[]>;
  /** Unfiled songs by date section, newest first; empty sections omitted. */
  byPeriod: { period: Period; jobs: AudioJob[] }[];
}

/**
 * Split jobs into project sections and date sections. `filing` maps a job
 * id to its project; a job filed under a project that no longer exists
 * (`projectIds` lacks it) is treated as unfiled rather than hidden.
 */
export function groupHistory(
  jobs: AudioJob[],
  filing: Map<string, string | null>,
  projectIds: Set<string>,
  now: Date,
): HistoryGroups {
  const sorted = [...jobs].sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id));
  const byProject = new Map<string, AudioJob[]>();
  const periods = new Map<Period, AudioJob[]>();
  for (const job of sorted) {
    const project = filing.get(job.id) ?? null;
    if (project !== null && projectIds.has(project)) {
      const list = byProject.get(project) ?? [];
      list.push(job);
      byProject.set(project, list);
    } else {
      const period = periodOf(job.created_at, now);
      const list = periods.get(period) ?? [];
      list.push(job);
      periods.set(period, list);
    }
  }
  return {
    byProject,
    byPeriod: PERIODS.filter((p) => periods.has(p)).map((p) => ({ period: p, jobs: periods.get(p)! })),
  };
}

/** Section keys for the persisted collapsed state. */
export const periodKey = (p: Period): string => `period:${p}`;
export const projectKey = (id: string): string => `project:${id}`;
