"""Python AST knowledge extractor — zero external dependencies.

Uses the stdlib ``ast`` module to extract classes, functions, imports,
inheritance, decorators, and call relationships from Python source files.
Produces compact knowledge summaries that save agent tokens compared to
reading raw source files.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..models import ExtractionResult, Relation


class PythonASTExtractor:
    """Extract knowledge from Python source files via stdlib AST parsing.

    Extracts:
    - Classes with base classes, decorators, and method lists
    - Functions/methods with decorators and call targets
    - Import relationships (import X, from X import Y)
    - Inheritance (class Foo(Bar) → Foo INHERITS Bar)
    - Call relationships (function A calls function B)
    - Module-level docstrings and class/function docstrings

    All extraction is deterministic and free (no LLM tokens).
    """

    def __init__(self, *, domain: str = "code", confidence: float = 0.9) -> None:
        self._domain = domain
        self._confidence = confidence

    def extract(self, path: Path) -> list[ExtractionResult]:
        """Extract knowledge from all ``.py`` files under *path*."""
        path = Path(path)
        if path.is_file():
            files = [path] if path.suffix == ".py" else []
        else:
            files = sorted(path.rglob("*.py"))

        results: list[ExtractionResult] = []
        for py_file in files:
            result = self._extract_file(py_file)
            if result is not None:
                results.append(result)
        return results

    def _extract_file(self, file_path: Path) -> ExtractionResult | None:
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return None

        visitor = _ASTVisitor(str(file_path))
        visitor.visit(tree)

        if not visitor.entities and not visitor.summaries:
            return None

        text = "\n".join(visitor.summaries)
        return ExtractionResult(
            text=text,
            source=str(file_path),
            entities=visitor.entities,
            relations=visitor.relations,
            confidence=self._confidence,
            domain=self._domain,
            tags=["ast-extracted", "python"],
            metadata={
                "language": "python",
                "source_file": str(file_path),
                "classes": visitor.class_names,
                "functions": visitor.function_names,
                "imports": visitor.import_names,
            },
        )


class _ASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.entities: list[str] = []
        self.relations: list[Relation] = []
        self.summaries: list[str] = []
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.import_names: list[str] = []
        self._current_class: str | None = None
        self._current_function: str | None = None
        self._module_name = Path(file_path).stem

    def visit_Module(self, node: ast.Module) -> None:
        docstring = ast.get_docstring(node)
        if docstring:
            self.summaries.append(f"Module {self._module_name}: {docstring.split(chr(10))[0]}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        name = node.name
        self.entities.append(name)
        self.class_names.append(name)

        parts = [f"Class {name}"]

        # Bases (inheritance)
        bases = []
        for base in node.bases:
            base_name = _name_of(base)
            if base_name:
                bases.append(base_name)
                self.entities.append(base_name)
                self.relations.append(Relation(name, "inherits", base_name))
        if bases:
            parts.append(f"inherits {', '.join(bases)}")

        # Decorators
        decorators = [_name_of(d) for d in node.decorator_list if _name_of(d)]
        if decorators:
            parts.append(f"decorated with @{', @'.join(decorators)}")

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            parts.append(f"— {docstring.split(chr(10))[0]}")

        # Methods
        methods = [n.name for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if methods:
            parts.append(f"methods: {', '.join(methods)}")

        self.summaries.append(". ".join(parts) + ".")

        prev_class = self._current_class
        self._current_class = name
        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._current_class:
            qualified = f"{self._current_class}.{node.name}"
        else:
            qualified = node.name

        self.entities.append(qualified)
        self.function_names.append(qualified)

        if self._current_class:
            self.relations.append(Relation(self._current_class, "has_method", qualified))

        parts = [f"{'Async function' if isinstance(node, ast.AsyncFunctionDef) else 'Function'} {qualified}"]

        # Decorators
        decorators = [_name_of(d) for d in node.decorator_list if _name_of(d)]
        if decorators:
            parts.append(f"decorated with @{', @'.join(decorators)}")

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            parts.append(f"— {docstring.split(chr(10))[0]}")

        # Parameters (skip self/cls)
        params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
        if params:
            parts.append(f"params: {', '.join(params)}")

        self.summaries.append(". ".join(parts) + ".")

        # Extract call targets
        prev_func = self._current_function
        self._current_function = qualified
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = _name_of(child.func)
                if call_name and call_name != qualified:
                    self.relations.append(Relation(qualified, "calls", call_name))
        self._current_function = prev_func

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            self.import_names.append(name)
            self.entities.append(name)
            self.relations.append(Relation(self._module_name, "imports", name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.import_names.append(full)
            self.entities.append(alias.name)
            self.relations.append(Relation(self._module_name, "imports", alias.name))


def _name_of(node: ast.expr) -> str | None:
    """Best-effort name extraction from an AST expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name_of(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return None
