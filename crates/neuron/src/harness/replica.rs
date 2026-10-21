//! Copies of one model on several devices (#397).
//!
//! A mining rig carries many identical small cards on PCIe x1 risers,
//! where tensor parallelism is not viable: the throughput shape is one
//! independent copy of the model per card. The neuron keeps those copies
//! behind one model id, so cortex sees a single model whose capacity is
//! their sum.
//!
//! Admission stays per *model*: one [`AdmissionController`] with an
//! in-flight slot per replica, so the queue, the per-principal fair share
//! and the anonymous ceiling all see the model's whole capacity. An
//! admitted request then takes an idle replica from a [`ReplicaPool`].
//! Admission never lets in more requests than there are replicas, so a
//! replica is free whenever a request reaches the pool, unless one has
//! failed.
//!
//! Idle replicas are handed out round-robin (a returned replica goes to
//! the back of the line), which spreads heat and wear across the cards
//! instead of working the first one hardest.
//!
//! [`AdmissionController`]: super::admission::AdmissionController

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// The idle set of a replicated model's copies.
pub struct ReplicaPool {
    idle: Mutex<VecDeque<usize>>,
    /// One permit per idle, healthy replica.
    available: Semaphore,
    healthy: Vec<AtomicBool>,
    busy: Vec<AtomicUsize>,
}

impl ReplicaPool {
    pub fn new(replicas: usize) -> Arc<Self> {
        Arc::new(Self {
            idle: Mutex::new((0..replicas).collect()),
            available: Semaphore::new(replicas),
            healthy: (0..replicas).map(|_| AtomicBool::new(true)).collect(),
            busy: (0..replicas).map(|_| AtomicUsize::new(0)).collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.healthy.len()
    }

    pub fn is_empty(&self) -> bool {
        self.healthy.is_empty()
    }

    /// Wait for an idle replica. `None` once every replica has failed.
    pub async fn checkout(self: &Arc<Self>) -> Option<ReplicaLease> {
        let permit = self.available.acquire().await.ok()?;
        permit.forget();
        let index = self
            .idle
            .lock()
            .expect("replica pool poisoned")
            .pop_front()?;
        self.busy[index].fetch_add(1, Ordering::AcqRel);
        Some(ReplicaLease {
            pool: Arc::clone(self),
            index,
        })
    }

    /// Requests replica `index` is running.
    pub fn in_flight(&self, index: usize) -> usize {
        self.busy[index].load(Ordering::Acquire)
    }

    pub fn is_healthy(&self, index: usize) -> bool {
        self.healthy[index].load(Ordering::Acquire)
    }

    pub fn healthy_count(&self) -> usize {
        self.healthy
            .iter()
            .filter(|h| h.load(Ordering::Acquire))
            .count()
    }

    /// Take a replica out of service for good. It is not handed out again;
    /// when it was the last one, waiters are woken with `None`.
    fn retire(&self, index: usize) {
        self.healthy[index].store(false, Ordering::Release);
        if self.healthy_count() == 0 {
            self.available.close();
        }
    }
}

/// A replica held for one request. Dropping it returns the replica to the
/// pool, unless it was marked failed.
pub struct ReplicaLease {
    pool: Arc<ReplicaPool>,
    index: usize,
}

impl ReplicaLease {
    pub fn index(&self) -> usize {
        self.index
    }

    /// The replica's device failed: never hand it out again.
    pub fn mark_failed(&self) {
        self.pool.retire(self.index);
    }
}

impl Drop for ReplicaLease {
    fn drop(&mut self) {
        self.pool.busy[self.index].fetch_sub(1, Ordering::AcqRel);
        if self.pool.is_healthy(self.index) {
            self.pool
                .idle
                .lock()
                .expect("replica pool poisoned")
                .push_back(self.index);
            self.pool.available.add_permits(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdmissionConfig;
    use crate::harness::admission::AdmissionController;
    use std::time::Duration;

    /// #397's acceptance: eight replicas, sixteen concurrent requests. Every
    /// replica works, none ever runs two at once, and nothing is refused.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sixteen_requests_spread_over_eight_replicas_one_at_a_time() {
        const REPLICAS: usize = 8;
        let pool = ReplicaPool::new(REPLICAS);
        let admission = Arc::new(AdmissionController::new(&AdmissionConfig {
            max_in_flight: REPLICAS,
            max_queue_depth: 16,
            max_per_principal: 0,
            ..AdmissionConfig::default()
        }));
        let peak: Arc<Vec<AtomicUsize>> =
            Arc::new((0..REPLICAS).map(|_| AtomicUsize::new(0)).collect());
        let served: Arc<Vec<AtomicUsize>> =
            Arc::new((0..REPLICAS).map(|_| AtomicUsize::new(0)).collect());

        let tasks: Vec<_> = (0..16)
            .map(|i| {
                let (pool, admission) = (Arc::clone(&pool), Arc::clone(&admission));
                let (peak, served) = (Arc::clone(&peak), Arc::clone(&served));
                tokio::spawn(async move {
                    let principal = format!("caller-{i}");
                    let _permit = admission.enter(Some(&principal)).await.expect("admitted");
                    let lease = pool.checkout().await.expect("a replica");
                    let running = pool.in_flight(lease.index());
                    peak[lease.index()].fetch_max(running, Ordering::AcqRel);
                    served[lease.index()].fetch_add(1, Ordering::AcqRel);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                })
            })
            .collect();
        for t in tasks {
            t.await.expect("request task");
        }

        for r in 0..REPLICAS {
            assert_eq!(peak[r].load(Ordering::Acquire), 1, "replica {r} peak");
            assert!(
                served[r].load(Ordering::Acquire) >= 1,
                "replica {r} never served"
            );
        }
        let total: usize = served.iter().map(|s| s.load(Ordering::Acquire)).sum();
        assert_eq!(total, 16);
        assert_eq!(admission.in_flight(), 0);
    }

    #[tokio::test]
    async fn idle_replicas_are_handed_out_round_robin() {
        let pool = ReplicaPool::new(3);
        let mut order = Vec::new();
        for _ in 0..6 {
            let lease = pool.checkout().await.expect("a replica");
            order.push(lease.index());
        }
        assert_eq!(order, [0, 1, 2, 0, 1, 2]);
    }

    #[tokio::test]
    async fn a_failed_replica_is_never_handed_out_again() {
        let pool = ReplicaPool::new(2);
        let lease = pool.checkout().await.expect("a replica");
        assert_eq!(lease.index(), 0);
        lease.mark_failed();
        drop(lease);
        for _ in 0..4 {
            assert_eq!(pool.checkout().await.expect("a replica").index(), 1);
        }
        assert_eq!(pool.healthy_count(), 1);
    }

    #[tokio::test]
    async fn waiters_wake_when_the_last_replica_fails() {
        let pool = ReplicaPool::new(1);
        let lease = pool.checkout().await.expect("a replica");
        let waiter = {
            let pool = Arc::clone(&pool);
            tokio::spawn(async move { pool.checkout().await.is_none() })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        lease.mark_failed();
        drop(lease);
        let woke_empty = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("waiter woke")
            .expect("waiter task");
        assert!(woke_empty);
    }
}
