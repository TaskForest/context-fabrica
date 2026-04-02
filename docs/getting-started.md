# Getting Started

## 1. Install the package

```bash
python -m pip install .
python -m pip install -r requirements-v2.txt
```

## 2. Bootstrap the project

```bash
context-fabrica-bootstrap --dsn "postgresql:///context_fabrica"
```

## 3. Verify the runtime

```bash
context-fabrica-doctor --dsn "postgresql:///context_fabrica"
```

## 4. Run the demo

```bash
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project
```

## 5. Inspect the projector queue

```bash
context-fabrica-projector --status
```
