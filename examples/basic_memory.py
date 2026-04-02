from __future__ import annotations

from context_fabrica.engine import DomainMemoryEngine


def main() -> None:
    engine = DomainMemoryEngine()
    engine.ingest(
        "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
        source="design-doc",
        domain="fintech",
        confidence=0.9,
        tags=["design-doc"],
    )
    engine.ingest(
        "LedgerAdapter writes transactions to event store.",
        source="runbook",
        domain="fintech",
        confidence=0.85,
        tags=["runbook"],
    )

    for hit in engine.query("How does PaymentsService interact with LedgerAdapter?", top_k=3):
        print(hit.record.record_id, round(hit.score, 3), hit.rationale)


if __name__ == "__main__":
    main()
