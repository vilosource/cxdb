// Copyright 2025 StrongDM Inc
// SPDX-License-Identifier: Apache-2.0

use blake3::Hasher;
use cxdb_server::store::Store;
use rmpv::Value;
use tempfile::tempdir;

#[test]
fn append_and_fork() {
    let dir = tempdir().expect("tempdir");
    let mut store = Store::open(dir.path()).expect("open store");

    let ctx = store.create_context(0).expect("create context");
    assert_eq!(ctx.head_turn_id, 0);

    let payload = b"hello world".to_vec();
    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let hash = hasher.finalize();

    let (first, _metadata) = store
        .append_turn(
            ctx.context_id,
            0,
            "com.example.Test".to_string(),
            1,
            1,
            0,
            payload.len() as u32,
            *hash.as_bytes(),
            &payload,
        )
        .expect("append first");

    let fork = store.fork_context(first.turn_id).expect("fork context");

    let second_payload = b"hello world".to_vec();
    let mut hasher2 = Hasher::new();
    hasher2.update(&second_payload);
    let hash2 = hasher2.finalize();

    let _second = store
        .append_turn(
            fork.context_id,
            0,
            "com.example.Test".to_string(),
            1,
            1,
            0,
            second_payload.len() as u32,
            *hash2.as_bytes(),
            &second_payload,
        )
        .expect("append second");

    assert!(store.blob_store.contains(hash.as_bytes()));

    let last = store.get_last(fork.context_id, 10, true).expect("get last");
    assert_eq!(last.len(), 2);
    assert_eq!(last[0].record.turn_id, first.turn_id);
}

#[test]
fn data_persists_across_reopen() {
    let dir = tempdir().expect("tempdir");

    let payload = b"persist me".to_vec();
    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let hash = hasher.finalize();

    let (context_id, turn_id) = {
        let mut store = Store::open(dir.path()).expect("open store");
        let ctx = store.create_context(0).expect("create context");
        let (turn, _meta) = store
            .append_turn(
                ctx.context_id,
                0,
                "com.example.Persist".to_string(),
                1,
                1,
                0,
                payload.len() as u32,
                *hash.as_bytes(),
                &payload,
            )
            .expect("append turn");
        (ctx.context_id, turn.turn_id)
    }; // store dropped, files closed

    // Reopen the same directory — data should still be there.
    let store = Store::open(dir.path()).expect("reopen store");
    let contexts = store.list_recent_contexts(100);
    assert!(
        !contexts.is_empty(),
        "expected at least one context after reopen"
    );
    let last = store
        .get_last(context_id, 10, true)
        .expect("get last after reopen");
    assert_eq!(last.len(), 1, "expected one turn after reopen");
    assert_eq!(last[0].record.turn_id, turn_id);
    assert!(
        store.blob_store.contains(hash.as_bytes()),
        "blob should persist after reopen"
    );
}

#[test]
fn indexes_parent_child_context_lineage() {
    let dir = tempdir().expect("tempdir");
    let mut store = Store::open(dir.path()).expect("open store");

    let parent = store.create_context(0).expect("create parent");
    let child = store.create_context(0).expect("create child");
    let grandchild = store.create_context(0).expect("create grandchild");

    let child_payload =
        encode_context_metadata_payload(Some(parent.context_id), Some(parent.context_id));
    let child_hash = blake3::hash(&child_payload);
    store
        .append_turn(
            child.context_id,
            0,
            "cxdb.ConversationItem".to_string(),
            1,
            1,
            0,
            child_payload.len() as u32,
            *child_hash.as_bytes(),
            &child_payload,
        )
        .expect("append child first turn");

    let grandchild_payload =
        encode_context_metadata_payload(Some(child.context_id), Some(parent.context_id));
    let grandchild_hash = blake3::hash(&grandchild_payload);
    store
        .append_turn(
            grandchild.context_id,
            0,
            "cxdb.ConversationItem".to_string(),
            1,
            1,
            0,
            grandchild_payload.len() as u32,
            *grandchild_hash.as_bytes(),
            &grandchild_payload,
        )
        .expect("append grandchild first turn");

    let direct_children = store.child_context_ids(parent.context_id);
    assert_eq!(direct_children, vec![child.context_id]);

    let descendants = store.descendant_context_ids(parent.context_id, None);
    assert_eq!(descendants, vec![grandchild.context_id, child.context_id]);
}

/// vafi#38 — `list_contexts_by_label` must return only contexts whose
/// labels include the exact filter string, ordered by recency, truncated
/// to `limit`. Closes the bug where `GET /v1/contexts?task_id=X` silently
/// returned the global most-recent list because the handler never consulted
/// the label index.
#[test]
fn list_contexts_by_label_filters_by_exact_label() {
    let dir = tempdir().expect("tempdir");
    let mut store = Store::open(dir.path()).expect("open store");

    let ctx_a1 = store.create_context(0).expect("create A1");
    let ctx_b = store.create_context(0).expect("create B");
    let ctx_a2 = store.create_context(0).expect("create A2");
    let ctx_unlabeled = store.create_context(0).expect("create unlabeled");

    // Two contexts labeled task:A, one labeled task:B, one with no labels.
    // ctx_a2 is created last → has the higher created_at_unix_ms, so it
    // should sort first.
    append_first_turn_with_labels(&mut store, ctx_a1.context_id, &["task:A"]);
    append_first_turn_with_labels(&mut store, ctx_b.context_id, &["task:B"]);
    append_first_turn_with_labels(&mut store, ctx_a2.context_id, &["task:A"]);
    // unlabeled context: append a turn with no labels at all.
    append_first_turn_with_labels(&mut store, ctx_unlabeled.context_id, &[]);

    let task_a = store.list_contexts_by_label("task:A", 50);
    let ids: Vec<u64> = task_a.iter().map(|h| h.context_id).collect();
    assert_eq!(
        ids,
        vec![ctx_a2.context_id, ctx_a1.context_id],
        "task:A filter must return only the two task:A contexts, newest first"
    );

    let task_b = store.list_contexts_by_label("task:B", 50);
    let ids: Vec<u64> = task_b.iter().map(|h| h.context_id).collect();
    assert_eq!(ids, vec![ctx_b.context_id]);

    // No match → empty.
    let none = store.list_contexts_by_label("task:DOES-NOT-EXIST", 50);
    assert!(none.is_empty());

    // Limit truncates after sort.
    let limited = store.list_contexts_by_label("task:A", 1);
    let ids: Vec<u64> = limited.iter().map(|h| h.context_id).collect();
    assert_eq!(ids, vec![ctx_a2.context_id]);
}

fn append_first_turn_with_labels(
    store: &mut Store,
    context_id: u64,
    labels: &[&str],
) {
    let payload = encode_context_metadata_with_labels(labels);
    let hash = blake3::hash(&payload);
    store
        .append_turn(
            context_id,
            0,
            "cxdb.ConversationItem".to_string(),
            1,
            1,
            0,
            payload.len() as u32,
            *hash.as_bytes(),
            &payload,
        )
        .expect("append labeled first turn");
}

fn encode_context_metadata_with_labels(labels: &[&str]) -> Vec<u8> {
    // Wire format per server/src/store.rs extract_context_metadata:
    //   root: Map { 30 -> context_metadata }
    //   context_metadata: Map { 1 -> client_tag, 3 -> [labels...] }
    let label_values: Vec<Value> = labels.iter().map(|s| Value::from(*s)).collect();
    let mut meta_entries: Vec<(Value, Value)> = vec![
        (Value::from(1), Value::from("test-client")),
    ];
    if !labels.is_empty() {
        meta_entries.push((Value::from(3), Value::Array(label_values)));
    }
    let context_metadata = Value::Map(meta_entries);
    let root = Value::Map(vec![(Value::from(30), context_metadata)]);

    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &root).expect("encode payload");
    payload
}

fn encode_context_metadata_payload(
    parent_context_id: Option<u64>,
    root_context_id: Option<u64>,
) -> Vec<u8> {
    let mut provenance_entries = Vec::new();
    if let Some(parent) = parent_context_id {
        provenance_entries.push((Value::from(1), Value::from(parent)));
    }
    if let Some(root) = root_context_id {
        provenance_entries.push((Value::from(3), Value::from(root)));
    }

    let context_metadata = Value::Map(vec![
        (Value::from(1), Value::from("test-client")),
        (Value::from(10), Value::Map(provenance_entries)),
    ]);
    let root = Value::Map(vec![(Value::from(30), context_metadata)]);

    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &root).expect("encode payload");
    payload
}
